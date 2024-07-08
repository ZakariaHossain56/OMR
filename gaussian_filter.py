import math
import numpy as np

def gaussian(sigmax=1,sigmay=1):
    row = math.floor(5*sigmax)
    column=math.floor(5*sigmay)
    if(row%2==0):
        row+=1
    if (column % 2 == 0):
        column += 1
    #print(row)
    #print(column)
    centerx=row//2
    centery=column//2
    # print(centerx)
    # print(centery)
    constant=1/(2*math.pi*sigmax*sigmay)
    kernel = np.zeros((row, column))

    for x in range(row):
        for y in range(column):
            calc1=((x-centerx)**2)/(sigmax**2)+((y-centery)**2)/(sigmay**2)
            kernel[x][y] = constant * math.exp(-0.5 * calc1)

    minimum=np.min(kernel)
    normalized_kernel = kernel / minimum
    normalized_kernel = np.round(normalized_kernel).astype(np.uint8)

    # print(kernel)
    # print("Gaussian kernel")
    # print(normalized_kernel)
    return kernel


# gaussian(1,1)