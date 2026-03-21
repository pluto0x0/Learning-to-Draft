from tqdm.rich import tqdm
import time

for i in tqdm(enumerate(range(100000)), total=100000, desc="Processing"):
    time.sleep(0.0001)