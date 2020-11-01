import numpy as np

try:
    output_data = np.fromfile("first_pass_output.txt", dtype=float, sep='', offset=0)
    print(len(output_data))
    print(output_data[:30])
except Exception as e:
    print(e)