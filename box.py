#Originally from http://web.cse.ohio-state.edu/~crawfis.3/cse581/Slides/cse581_extra_Modeling.pdf
#  Now unsure (changed source due to bugs (that weren't the mesh's fault))

class Box:

    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]
    ] 
    indicies = [
        [0, 1, 2], [0, 2, 3], [2, 4, 3],
		[2, 5, 4], [1, 5, 2], [1, 6, 5],
		[0, 4, 7], [0, 3, 4], [5, 7, 4], 
		[5, 6, 7], [0, 7, 6], [0, 6, 1]
    ]
