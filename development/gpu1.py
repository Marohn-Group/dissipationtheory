# author: John A. Marohn (jam99@cornell.edu)
# date: 2025-09-14
# summary: Demonstrate GPU acceleration of matrix multiplication using PyTorch



# jam99 laptop AS-CHM-MARO-12 laptop
# Intel UHD Graphics 617
# The Intel UHD Graphics 617 is NOT supported for hardware acceleration with PyTorch :-(
# 
# jam99 AS-CHM-MARO-16 laptop 
# Apple M3 GPU and Metal 3 support
# PyTorch officially supports Apple Silicon M3 chips through the Metal Performance Shaders (MPS) backend :-)
#
# References:
#
#  100x Faster Than NumPy... (GPU Acceleration)
#  https://www.youtube.com/watch?v=Vw2xg1bYHkY
#
#  GPU-Accelerated Ideal Gas Law Simulation [part of a series of videos]
#  https://www.youtube.com/watch?v=2XckqFzUiYU
# 
#  Luke Polson
#  https://github.com/lukepolson/youtube_channel/tree/main/Python%20GPU
#
# I am running python 3.10.2
#
# $ conda install pytorch
# $ conda install torchvision <== not needed here
#
# CUDA requires an NVIDIA GPU, which is not available on Apple Silicon Macs.
# For Apple Silicon Macs, PyTorch uses the MPS backend for GPU acceleration.
#
# https://docs.pytorch.org/docs/stable/notes/mps.html

import torch
import numpy as np
import time
import platform
import psutil 

def test0():
    """Print out system information."""

    print("--- CPU Information ---")
    print(f"Processor: {platform.processor()}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Total cores: {psutil.cpu_count(logical=True)}")
    print(f"Current CPU frequency: {psutil.cpu_freq().current:.2f} MHz")

    print("\n--- PyTorch and GPU Information ---")
    print(f'PyTorch version: {torch.__version__}')
    print(f'It is {torch.cuda.is_available()} that CUDA is available')
    print(f'It is {torch.backends.mps.is_available()} that MPS is available')
    print(f'It is {torch.backends.mps.is_built()} that MPS is built')

def get_device(verbose=False):
    """Select the device to use for tensor computations.  
    Try CUDA/NVIDIA, then MPS, then CPU."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            if verbose:
                print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu") 
            if verbose:
                print("Using CPU")
    
    return device

def test1(n=8000, verbose=False):
    """Compare numpy and torch for single precision matrix multiplication."""

    if verbose:
        print('\n--- test1 ---')

    device = get_device(verbose=verbose)

    # Create tensors on the selected device

    x = torch.randn(n, n, device=device)
    y = torch.randn(n, n, device=device)

    if verbose:
        print(f'torch type {x.dtype}')

    # Perform a large matrix multipation using torch

    start = time.perf_counter()
    result1 = torch.matmul(x, y)
    finish = time.perf_counter()  
    duration1 = finish - start

    # Convert the tensors to numpy arrays

    x = x.cpu().numpy()  # x = np.random.randn(8000, 8000)
    y = y.cpu().numpy()  # y = np.random.randn(8000, 8000)

    if verbose:
        print(f'numpy type {x.dtype}')

    # Perform the same matrix multipation using numpy
    # Could also use result2 = x @ y

    start = time.perf_counter()
    result2 = np.matmul(x, y)
    finish = time.perf_counter()
    duration2 = finish - start

    # Compare the results

    diff = np.max(np.abs(result2-result1.cpu().numpy()))
    same = np.allclose(result1.cpu().numpy(), result2)

    # Print out the results

    if verbose:

        print(f'single-precision matrix multiplication with two {n} x {n} matrices')
        print(f'- performed tensor multiplication on {device} in {round(1e3 * duration1, 1)} ms')
        print(f'- performed numpy multiplication on CPU in {round(duration2, 2)} s')
        print(f'- according to np.allclose, are the results the same? {same}')
        print(f'- maximum difference between results: {diff:.1e}')
        print('')
        print(f'Torch was {round(duration2/duration1, 1)}x faster than numpy')

    return result1.device, n, duration1, duration2, diff, same

def test2(n=8000, verbose=False):
    """Compare numpy and torch for double precision matrix multiplication."""

    if verbose:
        print('\n--- test2 ---')

    device = get_device(verbose=verbose)

    # Create numpy arrays

    x = np.random.randn(n, n).astype(np.float64)
    y = np.random.randn(n, n).astype(np.float64)

    if verbose:
        print(f'numpy type is {x.dtype}')
   
    # Perform the matrix multipation using numpy
    # Could also use result2 = x @ y

    start = time.perf_counter()
    result2 = np.matmul(x, y)
    finish = time.perf_counter()
    duration2 = finish - start

    # Create tensors on the selected device

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    if verbose:
        print(f'torch type is {x.dtype}')

    # Perform a large matrix multipation using torch

    start = time.perf_counter()
    result1 = torch.matmul(x, y)
    finish = time.perf_counter()  
    duration1 = finish - start

    # Compare the results

    diff = np.max(np.abs(result2-result1.cpu().numpy()))
    same = np.allclose(result1.cpu().numpy(), result2)

    # Print out the results

    if verbose:

        print(f'double precision matrix multiplication with two {n} x {n} matrices')
        print(f'- performed tensor multiplication on {device} in {round(1e3 * duration1, 1)} ms')
        print(f'- performed numpy multiplication on CPU in {round(duration2, 2)} s')
        print(f'- according to np.allclose, are the results the same? {same}')
        print(f'- maximum difference between results: {diff:.1e}')
        print('')
        print(f'Torch was {round(duration2/duration1, 1)}x faster than numpy')

    return result1.device, n, duration1, duration2, diff, same

if __name__ == "__main__":   

    # Run the tests and printout the the results

    test0()
    device, n, duration1, duration2, diff, same = test1(8000, verbose=True)
    device, n, duration1, duration2, diff, same = test2(8000, verbose=True)


# Results for test1() on different machines:
#
#  jam99 laptop AS-CHM-MARO-12: 
#   False, False, False
#   using CPU
#   numpy time [s]: 27.1, 26.9, 27.2
#   tensor time [s]: 12.8, 12.2, 12.7
#   tensor fast than numpy by factor: 2.12, 2.21, 2.14 
#  
#  jam99 laptop AS-CHM-MARO-16: 
#   False, True, True
#   using MPS
#   numpy time [s]: 3.77, 3.84, 3.84
#   tensor time [s]: 0.036, 0.034, 0.035
#   tensor fast than numpy by factor: 104, 113, 109
#
# Tremendous speedup!
#
#  new laptop tensor vs old laptop numpy: 800x
#  new laptop tensor vs old laptop tensor: 360x
#
# Conclude that I should rewrite the key parts of dissipationtheory using torch and the GPU.
# (I should also rewrite fredemod.)
#
# And there is maybe another factor of 10x speedup possible using CPU threading; see multi3.py.
