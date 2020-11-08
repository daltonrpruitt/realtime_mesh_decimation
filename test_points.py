import utility
import pywavefront
import numpy as np

def test():
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    verts = [
                [5.241108,  4.804801,  5.1191983],
                [6.0089836, 6.2341,    5.86783  ],
                [5.2002597, 8.557337,  4.7756386],
                [5.8162093, 5.7044535, 5.6891794],
                [7.462566,  7.6431737, 7.4299254],
                [7.5428286, 7.48337,   7.511746 ],
                [6.0026984, 5.9301324, 6.0912952],
                [5.8856835, 5.946808,  6.0385847],
            ]

    xdata = [v[0] for v in verts]
    ydata = [v[1] for v in verts]
    zdata = [v[2] for v in verts]

    print(xdata, ydata, zdata)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')
    fig.show()
    input()



if __name__ == "__main__":
    test()