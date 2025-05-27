import torch

def test_gpu():
    print("\n=== Basic GPU Test ===\n")
    
    # 1. Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"1. CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if not cuda_available:
        print("\nCUDA is not available. Ending test.")
        return
    
    # 2. Get GPU details
    device_count = torch.cuda.device_count()
    print(f"\n2. Found {device_count} GPU device(s):")
    
    for i in range(device_count):
        print(f"\nDevice {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory/1024**3:.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    # 3. Simple tensor operations
    print("\n3. Running simple GPU calculations:")
    
    try:
        # Create tensors
        a = torch.randn(10000, 10000, device='cuda')
        b = torch.randn(10000, 10000, device='cuda')
        
        # Matrix multiplication (stress test)
        start_time = time.time()
        c = torch.matmul(a, b)
        elapsed = time.time() - start_time
        
        print(f"  Matrix multiplication completed in {elapsed:.4f} seconds")
        print(f"  Result checksum: {c.sum().item():.2f}")
        
        # Memory test
        del a, b, c
        torch.cuda.empty_cache()
        print("  Memory test passed")
        
    except Exception as e:
        print(f"  ❌ Calculation failed: {str(e)}")
        return
    
    # 4. CUDA device check
    print("\n4. Device selection test:")
    try:
        device = torch.device('cuda:0')
        x = torch.ones(5, device=device)
        print(f"  Created tensor on GPU: {x}")
        print(f"  Tensor device: {x.device}")
    except Exception as e:
        print(f"  ❌ Failed to create tensor on GPU: {str(e)}")
    
    print("\n=== GPU Test Complete ===")

if __name__ == "__main__":
    import time
    start_time = time.time()
    test_gpu()
    print(f"\nTotal test time: {time.time() - start_time:.2f} seconds")