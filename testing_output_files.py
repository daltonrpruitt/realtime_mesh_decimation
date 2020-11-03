import numpy as np
import os

try:
    print(os.path.getsize("./first_pass_output.npy"))
    output_data = np.load("first_pass_output.npy")
    #output_data = np.fromfile("first_pass_output.bin", dtype="f4", sep='', offset=0)
    print(output_data.shape)
    print(sum(sum(sum(output_data))))
except Exception as e:
    print(e)