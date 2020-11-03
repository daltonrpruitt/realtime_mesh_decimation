import os

import moderngl_window as mglw


class BasicWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    #title = "ModernGL Example"
    window_size = (80, 45)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4

    #resource_dir = os.path.normpath(os.path.join(__file__, '../data'))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)