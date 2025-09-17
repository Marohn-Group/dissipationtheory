# author: John A. Marohn (jam99@cornell.edu)
# date: 2025-09-14
# summary: Demonstrate parallel processing using the concurrent.futures module's ProcessPoolExecutor
#          Run 20 processes of 1 second each, using multiple CPU cores
#          Compare to single-threaded execution time of 20 seconds

import time  
import concurrent.futures
import numpy as np
import random

def compute(seconds):
    """A simple function that sleeps for a given number of seconds."""

    id = random.randint(1000, 9999)
    print(f'starting process {id} for {seconds} seconds')
    time.sleep(seconds)
    return f'  finish process {id} after {seconds} seconds'  

if __name__ == "__main__":   

    print('starting 20 processes of 1 second each...')
    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:

        secs = np.ones(20) 
        results = [executor.submit(compute, sec) for sec in secs]
        
        for f in concurrent.futures.as_completed(results):
            print(f.result())

    finish = time.perf_counter()  
    duration = finish - start

    print('\n--- summary ---')
    print(f'Using concurrent.futures.ProcessPoolExecutor')
    print(f'number of threads = {executor._max_workers}') 
    print(f'run time = {round(duration, 2)} seconds')
    print(f' speedup = {round(20/duration, 2)}x')
    print('---------------')
 