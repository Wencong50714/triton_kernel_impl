import torch
import triton
import triton.language as tl
from pathlib import Path


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """eager mode Softmax"""
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    sm_out = numerator / denominator[:, None]
    return sm_out


@triton.jit
def _softmax_kernel_online(
    x_ptr, sm_out_ptr, rows, cols, B0: tl.constexpr, B1: tl.constexpr
):
    # Program ID and row offsets
    pid = tl.program_id(0)
    start_row = pid * B0
    row_offsets = start_row + tl.arange(0, B0)

    # Use constant for faster log2(e)
    log2_e = 1.44269504

    # Pre-compute row offsets
    row_mask = row_offsets < rows

    # Initialize with proper masking
    x_max = tl.where(row_mask, tl.full((B0,), -float("inf"), dtype=tl.float32), 0.0)
    d_i = tl.zeros((B0,), dtype=tl.float32)

    # First pass: find max and compute exponential sum
    for j in range(0, cols, B1):
        # Setup column indices and masks
        col_offsets = tl.arange(0, B1)
        col_mask = col_offsets < (cols - j)

        # Create 2D mask grid
        mask = row_mask[:, None] & col_mask[None, :]

        # Compute memory offsets once
        row_ptrs = x_ptr + row_offsets[:, None] * cols
        ptrs = row_ptrs + (j + col_offsets[None, :])

        # Load data with mask
        x = tl.load(ptrs, mask=mask, other=-float("inf"))

        # Compute max more efficiently using tl.max
        row_max = tl.max(x, axis=1)
        cur_max = tl.maximum(x_max, row_max)

        # Scale using the differing max values
        scale_factor = tl.exp2(log2_e * (x_max - cur_max))
        x_exp = tl.exp2(log2_e * (x - cur_max[:, None]))
        d_i = scale_factor * d_i + tl.sum(x_exp, axis=1)
        x_max = cur_max

    # Second pass: normalize with the computed values
    for j in range(0, cols, B1):
        # Set up indices and masks
        col_offsets = tl.arange(0, B1)
        col_mask = col_offsets < (cols - j)
        mask = row_mask[:, None] & col_mask[None, :]

        # Compute memory offsets efficiently
        row_ptrs = x_ptr + row_offsets[:, None] * cols
        x_ptrs = row_ptrs + (j + col_offsets[None, :])

        output_row_ptrs = sm_out_ptr + row_offsets[:, None] * cols
        output_ptrs = output_row_ptrs + (j + col_offsets[None, :])

        # Load values
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))

        # Normalize in one step
        z = tl.exp2(log2_e * (x - x_max[:, None])) / d_i[:, None]

        # Store results
        tl.store(output_ptrs, z, mask=mask)


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    assert x.dim() == 2, f"only accepts 2D tensors for now"

    # Adjust block sizes based on input dimensions
    # Better to use powers of 2 aligned with GPU architecture
    if cols <= 256:
        B1 = 128
    elif cols <= 1024:
        B1 = 256
    else:
        B1 = 512

    # Optimize number of warps based on block size
    num_warps = 4 if B1 <= 256 else 8 if B1 <= 512 else 16

    grid = (rows,)

    sm_out = torch.empty_like(x)
    _softmax_kernel_online[grid](
        x, sm_out, rows, cols, B0=1, B1=B1, num_warps=num_warps
    )
    return sm_out


@triton.jit
def _softmax_kernel_fused(
    output_ptr,
    stride_output_row,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,
):
    # setup input ptrs
    row_index = tl.program_id(0)

    row_start_ptr = input_ptr + (row_index * stride_input_row)
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    row_mask = col_offsets < num_cols

    # move to SRAM
    row = tl.load(input_pointers, mask=row_mask, other=float("-inf"))

    # softmax itself
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator

    # write back to HBM
    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=row_mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton impl of Softmax, fwd pass only"""
    rows, cols = x.shape
    assert x.dim() == 2, f"only accepts 2D tensors for now"
    block_size = triton.next_power_of_2(cols)
    num_warps = 4  # *32
    if block_size > 2047:  # 2048
        num_warps = 8
    if block_size > 4095:  # 4096
        num_warps = 16

    grid = (rows,)

    # allocate our output buffer
    sm_out = torch.empty_like(x)

    _softmax_kernel_fused[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps=num_warps,
    )

    return sm_out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            1024 * i for i in range(2, 50)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "fused",
            "online",
            "torch",
            "native",
        ],  # possible values for `line_arg``
        line_names=[
            "fused (Triton)",
            "online (Triton)",
            "torch",
            "native (torch)",
        ],  # label name for the lines
        styles=[
            ("blue", "-"),
            ("blue", "--"),
            ("green", "-"),
            ("green", "--"),
        ],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "fused":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_softmax(x), quantiles=quantiles
        )
    if provider == "online":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: online_softmax(x), quantiles=quantiles
        )
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, axis=-1), quantiles=quantiles
        )
    if provider == "native":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: naive_softmax(x), quantiles=quantiles
        )

    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # ======== Check Implementation Correctness
    M, N = 4096, 1024
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)

    torch_result = torch.softmax(x, dim=-1)
    optimized_result = online_softmax(x)

    op_correct = torch.allclose(torch_result, optimized_result, rtol=1e-3, atol=1e-3)
    print(f"Optimized Softmax result: {'✅' if op_correct else '❌'}")

    if op_correct:
        print("Running performance benchmark...")
        benchmark.run(show_plots=True, print_data=True, save_path=Path.cwd())
