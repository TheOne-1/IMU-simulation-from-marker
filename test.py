import time
import pandas as pd
import multiprocessing
from urllib.request import urlopen
import numpy as np

df = pd.DataFrame(np.zeros([1, 1]))
results = []


def processData(df):
    """Does some compute intensive operation on the data frame.
       Returns a list."""

    for i in range(2):
        df = df * 1.0
    return df.values.tolist()


def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process"""
    results.extend(result)


if __name__ == "__main__":
    start_time = time.time()

    # Repeats the compute intensive operation on 10 data frames concurrently
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for i in range(10):
        pool.apply_async(processData, args=(df,), callback=collect_results)
    pool.close()
    pool.join()

    # Converts list of lists to a data frame
    df = pd.DataFrame(results)
    print(df.shape)
    print("--- %s seconds ---" % (time.time() - start_time))