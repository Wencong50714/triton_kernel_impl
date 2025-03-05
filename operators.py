import triton
import triton.language as tl

from tensor_type import Float32, Int32

# ========================== Torch Spec ==========================


def flashatt_spec(
    q: Float32[200,], k: Float32[200,], v: Float32[200,]
) -> Float32[200,]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)


@triton.jit
def two_pass_flashatt_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504

    # Load Q
    off_i = tl.arange(0, B0)[:, None] + block_id_i * B0  # shape (B0, 1)
    mask_i = off_i < N0
    q = tl.load(q_ptr + off_i, mask=mask_i, other=0.0)  # shape (B0, 1)

    # intermediate variables
    x_max = tl.full((B0, 1), -float("inf"), dtype=tl.float32)
    d_i = tl.zeros((B0, 1), dtype=tl.float32)

    # First pass: find max and calculate denominator
    for j in range(0, T, B1):
        # Load K
        off_j = tl.arange(0, B1)[None, :] + j
        mask_j = off_j < T
        mask = mask_i & mask_j
        k = tl.load(k_ptr + off_j, mask=mask_j, other=0.0)  # shape (1, B1)

        # Calculate QK
        x = q * k  # shape (B0, B1)

        # Mask invalid positions with -inf before computing max
        x = tl.where(mask, x, float("-inf"))

        # Update max and denominators
        cur_max = tl.maximum(x_max, tl.max(x, axis=1, keep_dims=True))
        x_exp = tl.exp2(log2_e * (x - cur_max))

        # Apply mask again after exp (to get zeros)
        x_exp = tl.where(mask, x_exp, 0.0)

        # Update denominator with rescaling
        d_i = tl.exp2(log2_e * (x_max - cur_max)) * d_i + tl.sum(
            x_exp, axis=1, keep_dims=True
        )
        x_max = cur_max

    # Second pass: compute weighted sum
    o_i = tl.zeros((B0, 1), dtype=tl.float32)

    for j in range(0, T, B1):
        # Load K and V
        off_j = tl.arange(0, B1)[None, :] + j
        mask_j = off_j < T
        mask = mask_i & mask_j

        k = tl.load(k_ptr + off_j, mask=mask_j, other=0.0)  # shape (1, B1)
        v = tl.load(v_ptr + off_j, mask=mask_j, other=0.0)  # shape (1, B1)

        # Calculate QK and softmax
        x = q * k
        x = tl.where(mask, x, float("-inf"))

        x_exp = tl.exp2(log2_e * (x - x_max))
        x_exp = tl.where(mask, x_exp, 0.0)
        soft = x_exp / d_i

        # Compute weighted sum
        o_i += tl.sum(v * soft, axis=1, keep_dims=True)

    # Write back result
    tl.store(z_ptr + off_i, o_i, mask=mask_i)

    return


@triton.jit
def one_pass_flashatt_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504

    # Load Q
    off_i = tl.arange(0, B0)[:, None] + block_id_i * B0
    mask_i = off_i < N0
    q = tl.load(q_ptr + off_i, mask=mask_i, other=0.0)  # shape (B0, 1)

    # accumulator
    x_max = tl.full((B0, 1), -float("inf"), dtype=tl.float32)
    softmax_sum = tl.zeros((B0, 1), dtype=tl.float32)
    output = tl.zeros((B0, 1), dtype=tl.float32)

    for j in range(0, T, B1):
        # Load K & V
        off_j = tl.arange(0, B1)[None, :] + j
        mask_j = off_j < T
        mask = mask_i & mask_j

        k = tl.load(k_ptr + off_j, mask=mask_j, other=0.0)  # shape (1, B1)
        v = tl.load(v_ptr + off_j, mask=mask_j, other=0.0)  # shape (1, B1)

        # dot-product x
        x = q * k  # shape (B0, B1)
        x = tl.where(mask, x, float("-inf"))

        # Update max and denominators
        new_max = tl.maximum(x_max, tl.max(x, axis=1, keep_dims=True))
        exp_max_diff = tl.exp2(log2_e * (x_max - new_max))

        # calculate exp
        x_exp = tl.exp2(log2_e * (x - new_max))
        x_exp = tl.where(mask, x_exp, 0.0)

        # calculate new exp sum
        new_softmax_sum = exp_max_diff * softmax_sum + tl.sum(
            x_exp, axis=1, keep_dims=True
        )

        qkv_sum = tl.sum(v * x_exp, axis=1, keep_dims=True) / new_softmax_sum
        output = tl.fma(output, exp_max_diff * softmax_sum / new_softmax_sum, qkv_sum)

        # Update Accumulator
        x_max = new_max
        softmax_sum = new_softmax_sum

    tl.store(z_ptr + off_i, output, mask=mask_i)
    return
