import numpy as np
import os

try:
    print(os.path.getsize("./first_pass_output_pix.npy"))
    output_data = np.load("first_pass_output_pix.npy")
    #output_data = np.fromfile("first_pass_output.bin", dtype="f4", sep='', offset=0)
    print(output_data.shape)
    #print(sum(output_data[0::4])+sum(output_data[1::4])+sum(output_data[2::4]))
    print(output_data[20:25,15:20:,:])
    print(sum(sum(output_data)))
except Exception as e:
    print(e)