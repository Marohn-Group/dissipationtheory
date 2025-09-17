# author: John A. Marohn (jam99@cornell.edu)
# date: 2025-09-14
# summary: Demonstrate parallel processing using the concurrent.futures module's ThreadPoolExecutor
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

    with concurrent.futures.ThreadPoolExecutor() as executor: # <== changed to ThreadPoolExecutor from ProcessPoolExecutor

        secs = np.ones(20) 
        results = [executor.submit(compute, sec) for sec in secs]
        
        for f in concurrent.futures.as_completed(results):
            print(f.result())

    finish = time.perf_counter()  
    duration = finish - start

    print('\n--- summary ---')
    print(f'Using concurrent.futures.ThreadPoolExecutor')
    print(f'number of threads = {executor._max_workers}') 
    print(f'run time = {round(duration, 2)} seconds')
    print(f' speedup = {round(20/duration, 2)}x')
    print('---------------')

# Unlocking your CPU cores in Python (multiprocessing)
# https://www.youtube.com/watch?v=X7vBbelRXn0

# Results on different machines:
#
#  jam99 laptop AS-CHM-MARO-12 laptop: 
#   faster by 6.62x, 6.63x, 6.63x, 
#
#  jam99 AS-CHM-MARO-16 laptop: 
#   faster by 9.94x, 9.93x, 9.9x
#
# Significant speedup!
#
# Threading has to be run on the comand line, not in a Jupyter notebook.