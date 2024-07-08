import numpy as np


def sobel():
    horizontal_filter = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]
    )
    vertical_filter = np.array(
        [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]
    )

    # print(horizontal_filter)
    # print(vertical_filter)
    return (horizontal_filter,vertical_filter)

# sobel()