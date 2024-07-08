import numpy as np
import cv2
import gaussian_filter

def normalization(img):
    img = img.copy()
    normalized_convoluted_image = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    output = np.round(normalized_convoluted_image).astype(np.uint8)
    return output



def convolution(img,kernel):

    kernel_height = int(len(kernel))
    kernel_width = int(len(kernel[0]))
    centerx = kernel_width // 2
    centery = kernel_height // 2
    kernel_center = (centerx,centery)
    # print(f"kernel height {kernel_height}")
    pad_top = int(centerx)
    pad_bottom = int(len(kernel) - centerx - 1)
    pad_left = int(centery)
    pad_right = int(len(kernel[0]) - centery - 1)
    # print(pad_top)

    bordered_image = cv2.copyMakeBorder(src=img, top=pad_top, bottom=pad_bottom, left=pad_left,
                                        right=pad_right, borderType=cv2.BORDER_CONSTANT)
    # print(bordered_image)
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
                    relative_kernelx = i + n
                    relative_kernely = j + n

                    relative_imagex = n - i
                    relative_imagey = n - j

                    actual_imagex = relative_imagex + image_start_x
                    actual_imagey = relative_imagey + image_start_y

                    kernel_value = kernel[relative_kernelx][relative_kernely]
                    image_value = bordered_image[actual_imagex][actual_imagey]
                    result += (kernel_value * image_value)
                output[x][y] = result
    out = output[kernel_center[0]:-kernel_height + kernel_center[0] + 1,
                    kernel_center[1]:-kernel_width + kernel_center[1] + 1]

    return out

# img = cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("Input grayscale image",img)
# cv2.waitKey(0)
# gaussian = gaussian_filter.gaussian(1,1)
# convoluted_image = convolution(img,gaussian)
# normalized_convoluted_image = cv2.normalize(convoluted_image, convoluted_image, 0, 255, cv2.NORM_MINMAX)
# output = np.round(normalized_convoluted_image).astype(np.uint8)
# cv2.imshow("Convoluted image",output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()