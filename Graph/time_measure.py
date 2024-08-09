import time
import os


def time_measure(func, model_name, dataset_name, phase):
    def wrapper(*args, **kwargs):
        # Create a directory to store the time measurements
        if not os.path.exists("time_measurements"):
            os.makedirs("time_measurements")
        # Create a file to store the time measurements
        file_name = f"time_measurements/{model_name}_{dataset_name}_{phase}.txt"
        # Measure the time taken by the function 10 times
        times = []
        for i in range(10):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            times.append(end-start)
        # Write the time measurements to the file
        with open(file_name, "w") as file:
            for t in times:
                file.write(str(t) + "\n")
        return result
    return wrapper
