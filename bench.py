from operators import *
import torch
import triton
import inspect
import time
from typing import Dict, Any, Callable, Annotated
from typing_extensions import TypeAlias

def benchmark(torch_spec, triton_ops: Dict[str, Callable], nelem={}, B={"B0": 32}, device="gpu") -> bool:
    """
    Benchmark Triton operations against their PyTorch specification.
    
    Args:
        torch_spec: PyTorch implementation to compare against
        triton_ops: Dictionary of Triton kernel implementations to benchmark
        nelem: Dictionary of problem dimensions
        B: Dictionary of block sizes
        device: Device to run on ("gpu" or "cpu")
    
    Returns:
        bool: True if all implementations match the reference, False otherwise
    """
    device = "cuda"
    
    # Handle block sizes based on dimensions
    B = dict(B)
    if "N1" in nelem and "B1" not in B:
        B["B1"] = 32
    if "N2" in nelem and "B2" not in B:
        B["B2"] = 32

    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Parse function signature to get argument types and dimensions
    signature = inspect.signature(torch_spec)
    args = {}
    for n, p in signature.parameters.items():
        args[n + "_ptr"] = (p.annotation.dims, p)
    args["z_ptr"] = (signature.return_annotation.dims, None)

    # Create random input tensors based on the signature
    tt_args = []
    for k, (v, t) in args.items():
        tt_args.append(torch.rand(*v, device=device) - 0.5)
        if t is not None and t.annotation.dtype == "int32":
            tt_args[-1] = torch.randint(-100000, 100000, v, device=device)

    # Define grid function for Triton kernels
    grid = lambda meta: (triton.cdiv(nelem["N0"], meta["B0"]),
                         triton.cdiv(nelem.get("N1", 1), meta.get("B1", 1)),
                         triton.cdiv(nelem.get("N2", 1), meta.get("B2", 1)))
    
    # Get reference output from torch_spec
    input_args = tt_args[:-1].copy()  # Copy input arguments without output tensor
    
    # Warmup torch implementation
    for _ in range(3):
        z_ref = torch_spec(*input_args)
    
    # Benchmark torch implementation
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    iterations = 10
    for _ in range(iterations):
        z_ref = torch_spec(*input_args)
    torch.cuda.synchronize() if device == "cuda" else None
    torch_time = (time.time() - start_time) / iterations
    
    # Print torch benchmark results
    print(f"\n{'=' * 50}")
    print(f"PyTorch implementation: {torch_time*1000:.4f} ms")
    print(f"{'=' * 50}")
    
    # Test each Triton implementation
    all_match = True
    
    for name, triton_op in triton_ops.items():
        # Create a copy of output tensor for each implementation
        z = torch.zeros_like(tt_args[-1])
        triton_args = input_args + [z]
        
        # Warmup
        for _ in range(3):
            triton_op[grid](*triton_args, **B, **nelem)
        
        # Benchmark
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        for _ in range(iterations):
            triton_op[grid](*triton_args, **B, **nelem)
        torch.cuda.synchronize() if device == "cuda" else None
        triton_time = (time.time() - start_time) / iterations
        
        # Check correctness
        match = torch.allclose(z, z_ref, rtol=1e-3, atol=1e-3)
        match_emoji = "✅" if match else "❌"
        all_match = all_match and match
        
        # Print results
        print(f"\nTriton implementation: {name}")
        print(f"Time: {triton_time*1000:.4f} ms ({torch_time/triton_time:.2f}x speedup)")
        print(f"{match_emoji} Results match: {match}")
        
        if not match:
            print(f"Inputs shape: {[arg.shape for arg in input_args]}")
            print(f"Yours: {z.dtype}, {z.shape}")
            print(f"Spec: {z_ref.dtype}, {z_ref.shape}")
            print("Diff (sample of mismatches):")
            mask = ~torch.isclose(z, z_ref, rtol=1e-3, atol=1e-3)
            if mask.any():
                indices = mask.nonzero(as_tuple=True)
                sample_size = min(5, indices[0].size(0))
                for i in range(sample_size):
                    idx = tuple(indices[j][i] for j in range(len(indices)))
                    print(f"  At {idx}: yours={z[idx].item():.6f}, ref={z_ref[idx].item():.6f}, diff={z[idx].item()-z_ref[idx].item():.6f}")
    
    return all_match

if __name__ == "__main__":
    # add your operator here

    # TEST flash
    print("===============flash attention operator===============")
    flash_att_ops = { "two_pass":two_pass_flashatt_kernel,  "one_pass":one_pass_flashatt_kernel}

    benchmark(
        flashatt_spec,
        flash_att_ops,
        nelem={"N0": 200, "T": 200},
        B={"B0": 64, "B1": 32},
    )
    print("=======================================================\n")