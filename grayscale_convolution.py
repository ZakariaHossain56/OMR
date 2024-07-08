import math

import cv2
import numpy as np

def convolution(img,kernel,center):
    # print(kernel)
    # print(type)
    centerx, centery = center
    # print(f"centerx {centerx}")
    # print(centery)
    # padding the input image based on the center position
    matrix = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])
    kernel_height = int(len(kernel))
    kernel_width = int(len(kernel[0]))
    # print(f"kernel height {kernel_height}")
    pad_top = int(centerx)
    pad_bottom = int(len(kernel) - centerx - 1)
    pad_left = int(centery)
    pad_right = int(len(kernel[0]) - centery - 1)
    # print(pad_top)

    bordered_image = cv2.copyMakeBorder(src=img, top=pad_top, bottom=pad_bottom, left=pad_left,
                                        right=pad_right, borderType=cv2.BORDER_CONSTANT)
    print(bordered_image)
    output = np.zeros_like(bordered_image, dtype='float32')
    padded_height, padded_width = bordered_image.shape  # output image height and width
    # print(f"padded height {padded_height}")
    for x in range(centerx, padded_height - (kernel_height - (centerx + 1))):
        for y in range(centery, padded_width - (kernel_width - (centery + 1))):
            # starting position of the image for the convolution operation(with the border)
            image_start_x = x - centerx
            image_start_y = y - centery
            result = 0
            n = kernel_width // 2
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    relative_kernelx = i + 1
                    relative_kernely = j + 1
                    relative_imagex = n - i
                    relative_imagey = n - j
                    actual_imagex = relative_imagex + image_start_x
                    actual_imagey = relative_imagey + image_start_y
                    kernel_value = kernel[relative_kernelx][relative_kernely]
                    image_value = bordered_image[actual_imagex][actual_imagey]
                    result += (kernel_value * image_value)
                output[x][y] = result
    print(output)

    # cv2.imshow('grayscaled input image', img)
    # cv2.waitKey(0)  # Wait until a key is pressed

    # cv2.imshow("convoluted output image", output)
    #
    # cv2.waitKey(0)  # Wait until a key is pressed
    # cv2.destroyAllWindows()
    return output

def grayscaleConvolution(kernel_with_type,center):
    img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Input grayscaled image', img)
    kernel, type = kernel_with_type
    if type == "sobel":
        horizontal_kernel, vertical_kernel = kernel
        # print("Horizontal sobel filter")
        # print(horizontal_kernel)
        # print("Vertical sobel filter")
        # print(vertical_kernel)
        horizontal_convolution=convolution(img,horizontal_kernel,center)
        cv2.imshow("Horizontal convolution",horizontal_convolution)
        cv2.waitKey(0)
        vertical_convolution = convolution(img, vertical_kernel, center)
        cv2.imshow("Vertical convolution", vertical_convolution)

        height, width = img.shape
        output = np.zeros_like(img, dtype='float32')

        for x in range(0, height):
            for y in range(0, width):
                dx = horizontal_convolution[x, y]
                dy = vertical_convolution[x, y]

                result = math.sqrt(dx ** 2 + dy ** 2)
                output[x, y] = result
        cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
        output = np.round(output).astype(np.uint8)
        cv2.waitKey(0)
        cv2.imshow("Convoluted image",output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        output = convolution(img, kernel, center)
        cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
        output = np.round(output).astype(np.uint8)
        cv2.waitKey(0)
        cv2.imshow("Convoluted image",output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

