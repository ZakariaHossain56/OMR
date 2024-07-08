import gaussian_filter
import numpy as np

def derivative():
    kernel = gaussian_filter.gaussian(sigmax=0.7, sigmay=0.7)
    size = len(kernel)

    x_derivative = np.zeros((size, size))
    y_derivative = np.zeros((size, size))

    min1 = 1e2
    min2 = 1e2

    cx = size // 2
    for x in range(size):
        for y in range(size):
            cal1 = -(x - cx) / (0.7 ** 2)
            cal2 = -(y - cx) / (0.7 ** 2)
            x_derivative[x, y] = cal1 * kernel[x, y]
            y_derivative[x, y] = cal2 * kernel[x, y]

            if x_derivative[x, y] != 0:
                min1 = min(abs(x_derivative[x, y]), min1)

            if y_derivative[x, y] != 0:
                min2 = min(abs(y_derivative[x, y]), min2)

    normalized_x_derivative = (x_derivative / min1).astype(int)
    normalized_y_derivative = (y_derivative / min2).astype(int)

    # print("actual value")
    # print(x_derivative)
    # print(y_derivative)
    #
    # print("normalized value")
    # print(normalized_x_derivative)     #sobel vertical kernel
    # print(normalized_y_derivative)     #sobel horizontal kernel
    return (x_derivative, y_derivative)
