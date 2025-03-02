import torch
import triton
import triton.language as tl
from pathlib import Path

def naive_softmax(x: torch.Tensor)-> torch.Tensor:
    """ eager mode Softmax"""
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    sm_out = numerator/denominator[:,None]
    return sm_out

@triton.jit
def _softmax_kernel_online(
    x_ptr, sm_out, rows, cols, B0: tl.constexpr, B1: tl.constexpr
):
    # Program ID and row offsets
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    off_i = tl.arange(0, B0)[:, None] + block_id_i * B0
    
    # Use masks consistently for better performance
    row_mask = off_i < rows
    
    # Accumulator (only initialize for valid rows)
    x_max = tl.full((B0,), -float('inf'), dtype=tl.float32)
    d_i = tl.zeros((B0,), dtype=tl.float32)

    # First pass: find max and compute exponential sum
    for j in range(0, cols, B1):
        off_j = tl.arange(0, B1)[None, :] + j
        col_mask = off_j < cols
        mask = row_mask & col_mask
        
        # Compute offsets and load data with mask
        off_ij = off_i * cols + off_j
        x = tl.load(x_ptr + off_ij, mask, other=-float('inf'))
        
        # Update max values efficiently
        cur_max = tl.maximum(x_max, tl.max(x, axis=1))
        
        # Scale by the difference in max values
        x_exp = tl.exp2(log2_e * (x - cur_max[:, None]))
        d_i = tl.exp2(log2_e * (x_max - cur_max)) * d_i + tl.sum(x_exp, axis=1)
        x_max = cur_max

    # Second pass: normalize with the computed values
    for j in range(0, cols, B1):
        off_j = tl.arange(0, B1)[None, :] + j
        col_mask = off_j < cols
        mask = row_mask & col_mask
        
        off_ij = off_i * cols + off_j
        x = tl.load(x_ptr + off_ij, mask, other=-float('inf'))
        
        # Compute normalized values
        x_exp = tl.exp2(log2_e * (x - x_max[:, None]))
        z = x_exp / d_i[:, None]
        
        # Store results with proper masking
        tl.store(sm_out + off_ij, z, mask)


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    assert x.dim() == 2, f"only accepts 2D tensors for now"

    # Adjust block sizes based on input dimensions
    # Better to use powers of 2 aligned with GPU architecture
    if cols <= 256:
        B1 = 256
    elif cols <= 1024:
        B1 = 512
    else:
        B1 = 1024
    
    # For row blocks, adapt based on the matrix shape
    if rows <= 1024:
        B0 = 64
    else:
        B0 = 128
    
    # Optimize number of warps based on block size
    num_warps = 4 if B1 <= 256 else 8 if B1 <= 512 else 16
    
    grid = (triton.cdiv(rows, B0),)
    
    sm_out = torch.empty_like(x)
    _softmax_kernel_online[grid](x, sm_out, rows, cols, B0=B0, B1=B1, num_warps=num_warps)
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
    col_offsets = tl.arange(0,block_size)
    input_pointers = row_start_ptr + col_offsets

    row_mask = col_offsets < num_cols

    # move to SRAM
    row = tl.load(input_pointers,mask = row_mask, other = float("-inf") )

    # softmax itself
    safe_row = row - tl.max(row, axis=0) 
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator

    # write back to HBM
    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask= row_mask)



def fused_softmax(x:torch.Tensor)->torch.Tensor:
    """ Triton impl of Softmax, fwd pass only """
    rows, cols = x.shape
    assert x.dim() ==2, f"only accepts 2D tensors for now"
    block_size = triton.next_power_of_2(cols)
    num_warps = 4  # *32 
    if block_size > 2047: # 2048
        num_warps = 8
    if block_size > 4095: # 4096
        num_warps=16
    
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
        num_warps =num_warps

    )

    return sm_out

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'fused',
            'online',
            'torch',
            'native',
        ],  # possible values for `line_arg``
        line_names=[
            "fused (Triton)",
            "online (Triton)",
            "torch",
            "native (torch)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('blue', '--'), ('green', '-'), ('green', '--')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)


def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'fused':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_softmax(x), quantiles=quantiles)
    if provider == 'online':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: online_softmax(x), quantiles=quantiles)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)


    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def memory_scan():
    M, N = 4096, 1024
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)

    for name, func in [
        ("PyTorch softmax", lambda x: torch.softmax(x, dim=-1)),
        ("Online softmax", online_softmax),
        ("Fused softmax", fused_softmax),
        ("Naive softmax", naive_softmax)
    ]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        result = func(x)
        torch.cuda.synchronize()
        
        total_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"{name}: Current memory: {total_mem:.2f} MB, Peak memory: {peak_mem:.2f} MB")

if __name__ == "__main__":
    # ======== Check Implementation Correctness
    M, N = 4096, 1024
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)

    torch_result = torch.softmax(x, dim=-1)
    optimized_result = online_softmax(x)

    op_correct = torch.allclose(torch_result, optimized_result, rtol=1e-3, atol=1e-3)
    print(f"Optimized Softmax result: {'✅' if op_correct else '❌'}")

    if op_correct:
        print("Running performance benchmark...")
        benchmark.run(show_plots=True, print_data=True, save_path=Path.cwd())
        
        print("Running memory usage benchmark...")
        benchmark_memory.run(show_plots=True, print_data=True, save_path=Path.cwd())