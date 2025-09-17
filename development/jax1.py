# Author: John A. Marohn (jam99@cornell.edu)
# Date: 2025-09-16
#
# Summary: Demonstrate GPU acceleration of matrix multiplication using JAX
#          Compare to numpy on the CPU
#
# This code requires jax.
# Installed jax as follows on my Apple M3 laptop:
#
#    $ python3 -m venv ~/jax-metal
#    $ source ~/jax-metal/bin/activate
#    $ python -m pip install numpy wheel
#    $ python -m pip install jax-metal==0.1.1 jaxlib==0.5.0 jax==0.5.0
#
# Tested that jax is installed and working:
#
#    $ python -c 'import jax; print(jax.numpy.arange(10))'
#

import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time
import logging

def test1a(n=8000):
    """Create, multiply two large single-precision random matrices using numpy at the CPU.
    Time the execution."""

    x = np.random.randn(n, n).astype(np.float32)
    y = np.random.randn(n, n).astype(np.float32)

    start = time.perf_counter()

    result = np.matmul(x, y)
    
    finish = time.perf_counter()
    duration = finish - start

    return n, x, y, result, duration

def test1b(x, y):
    """Multiply two large single-precision random matrices using JAX and the GPU.
    Time the execution."""

    start = time.perf_counter()
    
    result = jnp.matmul(x, y)
    
    finish = time.perf_counter()
    duration = finish - start

    return result, duration

def test2(x, y):
    """Multiply two large single-precision random matrices using JAX and the GPU."""

    return jnp.matmul(x, y)

def compare_results(n, result1, duration1, result2, duration2):
    """Compare the results of numpy and jax matrix multiplication."""

    diff = np.max(np.abs(result1-np.array(result2)))
    same = np.allclose(result1, np.array(result2))

    print(f'single precision matrix multiplication with two {n} x {n} matrices')
    print(f'- performed numpy multiplication in {round(duration1, 2)} s')
    print(f'- performed jax multiplication in {round(1e3 * duration2, 1)} ms')
    print(f'- according to np.allclose, are the results the same? {same}')
    print(f'- maximum difference between results: {diff:.1e}')
    print(f'jax was {round(duration1/duration2, 1)}x faster than numpy')

if __name__ == "__main__":   

    # Try to turn off JAX info messages
    # (this doesn't seem to work)

    logger = logging.getLogger("jax._src.xla_bridge")
    logger.setLevel(logging.ERROR)

    # Run the tests and printout the the results

    print('--- test 1 ---')

    n, x1, x2, result1, duration1 = test1a(8000)  # create two random matrices
    result2, duration2 = test1b(x1, x2)           # pass the matrices to jax for multiplication
    compare_results(n, result1, duration1, result2, duration2)

    print('--- test 2 ---')

    n, x1, x2, result1, duration1 = test1a(8000)
    result2, duration2 = test1b(x1, x2)
    compare_results(n, result1, duration1, result2, duration2)

    print('--- test 3 ---')

    test2_jit = jax.jit(test2)  # complile the function
    result2 = test2_jit(x1, x2) # the first call may slower due to compilation

    n, x1, x2, result1, duration1 = test1a(8000)
    
    start = time.perf_counter()
    result2 = test2_jit(x1, x2) # the second call should be fast
    finish = time.perf_counter()
    duration = finish - start
    
    compare_results(n, result1, duration1, result2, duration2)

