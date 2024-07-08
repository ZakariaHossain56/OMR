from collections import deque

import cv2
import numpy as np


def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    if rowsAvailable:
        # Find the width and height of the images to resize to
        firstImgShape = imgArray[0][0].shape
        width = int(firstImgShape[1] * scale)
        height = int(firstImgShape[0] * scale)

        images = []
        for x in range(rows):
            col_images = []
            for y in range(cols):
                img = imgArray[x][y]
                if len(img.shape) == 2:  # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img, (width, height))
                col_images.append(img)
            images.append(np.hstack(col_images))
        ver = np.vstack(images)
    else:
        firstImgShape = imgArray[0].shape
        width = int(firstImgShape[1] * scale)
        height = int(firstImgShape[0] * scale)

        images = [cv2.resize(img, (width, height)) for img in imgArray]
        ver = np.hstack(images)


def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        # print("Area : ",area)
        if(area>50):
            peri = cv2.arcLength(i,True)    # total length of this contour
            approx = cv2.approxPolyDP(i, 0.02*peri, True)   # approximation of corner points
            # print("Corner points : ",len(approx))
            if(len(approx) == 4):
                rectCon.append(i)
    rectCon = sorted(rectCon,key=cv2.contourArea,reverse=True)
    # print(rectCon)
    return rectCon

def getCornerPoints(contour):
    peri = cv2.arcLength(contour, True)  # total length of this contour
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # approximation of corner points
    return approx

def reorderPoints(points):
    points = points.reshape((6,2))
    newPoints = np.zeros((6,1,2),np.int32)
    add = points.sum(1)     # axis no. 1
    print("Points")
    print(points)
    print("Sum")
    print(add)              # origin point has the smallest sum

    newPoints[0] = points[np.argmin(add)]   #[0,0]
    newPoints[3] = points[np.argmax(add)]   #[w,h]

    difference = np.diff(points,axis=1)
    print("Difference")
    print(difference)
    newPoints[1] = points[np.argmin(difference)]    #[w,0]
    newPoints[2] = points[np.argmax(difference)]    #[0,h]
    print("New points")
    print(newPoints)
    return newPoints

def splitBoxes(img):
    img = img.copy()
    print(img.shape)
    rows = np.vsplit(img,5)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    return boxes

def showAnswers(img,myIndex,questions,answers,choices,grading):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)

    for x in range(0,questions):

        myAns = myIndex[x]
        cX = (myAns*secW) + secW//2
        cY = (x*secH) + secH//2

        if(grading[x] == 1):
            myColor = (0,255,0)
        else:
            myColor = (0,0,255)
            correctAns = answers[x]
            cv2.circle(img, ((correctAns*secW)+secW//2, (x*secH)+secH//2), 10, (0,255,0), cv2.FILLED)

        cv2.circle(img,(cX,cY),10,myColor,cv2.FILLED)
    return img

def countNonZeroPixel(img):
    img = img.copy()
    cnt = 0
    row,col = img.shape
    for i in range(row):
        for j in range(col):
            if(img[i,j]>0):
                cnt += 1
    return cnt


# def manual_find_contours(img):
#     """
#     Manually find contours in a binary image.
#     """
#     img_copy = img.copy()
#     height, width = img_copy.shape
#
#     contours = []
#
#     directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
#                   (1, 0), (1, -1), (0, -1), (-1, -1)]
#
#     def is_valid(x, y):
#         return 0 <= x < height and 0 <= y < width
#
#     for i in range(height):
#         for j in range(width):
#             if img_copy[i, j] == 255:
#                 contour = []
#                 stack = [(i, j)]
#
#                 while stack:
#                     x, y = stack.pop()
#                     if is_valid(x, y) and img_copy[x, y] == 255:
#                         contour.append((y, x))  # Store as (column, row)
#                         img_copy[x, y] = 0
#
#                         for d in directions:
#                             nx, ny = x + d[0], y + d[1]
#                             if is_valid(nx, ny) and img_copy[nx, ny] == 255:
#                                 stack.append((nx, ny))
#
#                 if contour:
#                     contours.append(np.array(contour, dtype=np.int32).reshape((-1, 1, 2)))
#
#     return contours


def manual_draw_contours(img, contours, color, thickness):
    """
    Manually draw contours on an image.
    """
    for contour in contours:
        y=contour[1] + 10
        x=contour[0] + 10
        cv2.circle(img, (y, x), radius=1, color=color, thickness=thickness)


def calculate_area(contour):
    """
    Calculate the area of a polygon using the Shoelace formula.
    """
    n = len(contour)
    area = 0.0
    for i in range(n):
        x1, y1 = contour[i][0]
        x2, y2 = contour[(i + 1) % n][0]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def find_corners(contour):
    corners = []
    max_x = 0
    min_x = 1e6
    max_y = 0
    min_y = 1e6
    for i in range(len(contour)):
        max_x = max(contour[i][0],max_x)
        min_x = min(contour[i][0],min_x)
        max_y = max(contour[i][1], max_y)
        min_y = min(contour[i][1], min_y)
    top_left = (min_x,max_y)
    top_right = (max_x,max_y)
    bottom_left = (min_x,min_y)
    bottom_right = (max_x,min_y)
    corners.append(top_left)
    corners.append(top_right)
    corners.append(bottom_left)
    corners.append(bottom_right)

    return corners

def calculate_perimeter(contour):
    """
    Calculate the perimeter (arc length) of a polygon.
    """
    n = len(contour)
    perimeter = 0.0
    for i in range(n):
        x1, y1 = contour[i][0]
        x2, y2 = contour[(i + 1) % n][0]
        perimeter += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return perimeter


def ramer_douglas_peucker(contour, epsilon):
    """
    Approximate a contour to a polygon using the Ramer-Douglas-Peucker algorithm.
    """

    def distance_point_to_line(px, py, x1, y1, x2, y2):
        return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    def rdp(points, epsilon):
        dmax = 0.0
        index = 0
        end = len(points)
        for i in range(1, end - 1):
            d = distance_point_to_line(points[i][0][0], points[i][0][1], points[0][0][0], points[0][0][1],
                                       points[-1][0][0], points[-1][0][1])
            if d > dmax:
                index = i
                dmax = d
        if dmax >= epsilon:
            results1 = rdp(points[:index + 1], epsilon)
            results2 = rdp(points[index:], epsilon)
            result = results1[:-1] + results2
        else:
            result = [points[0], points[-1]]
        return result

    return np.array(rdp(contour, epsilon), dtype=np.int32).reshape(-1, 1, 2)


def rectContours(contours):
    rectCon = []
    for contour in contours:
        area = calculate_area(contour)
        if area > 0:
            peri = calculate_perimeter(contour)
            approx = ramer_douglas_peucker(contour, 0.02 * peri)
            if len(approx) == 6:
                rectCon.append(contour)
    rectCon = sorted(rectCon, key=calculate_area, reverse=True)
    return rectCon

def getCornerPointss(contour):
    """
    Manually get the corner points of a contour.
    """
    peri = calculate_perimeter(contour)
    approx = ramer_douglas_peucker(contour, 0.02 * peri)
    return approx



def dfs(image, sp_x, sp_y, to_replace, replace_with):
    height, width = image.shape
    parent_map = {}
    length = 0
    last = None

    stack = [(sp_x, sp_y, 0)]
    parent_map[(sp_x, sp_y)] = None

    while stack:
        x, y, it = stack.pop()
        if image[x, y] != to_replace:
            continue

        image[x, y] = replace_with

        it += 1
        if it > length:
            length = it
            last = (x, y)

        indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        for dx, dy in indices:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and image[nx, ny] == to_replace:
                if (nx, ny) not in parent_map:  # Check if not already visited
                    parent_map[(nx, ny)] = (x, y)
                    stack.append((nx, ny, it))

    points = []
    while last is not None:
        points.append(last)
        last = parent_map[last]
    points.reverse()
    return points


def get_edge_points(image):
    image = image.copy()
    height, width = image.shape
    pad = 10
    image = image[pad:height - pad, pad:width - pad]
    height, width = image.shape
    contours = []

    visited = {}

    def bfs(sx, sy):
        nonlocal visited
        to_it = (sx, sy)

        while to_it != None:
            queue = deque()
            queue.append(to_it)

            to_it = None
            count = 0
            while queue:
                x, y = queue.popleft()
                if visited.get((x, y)) == True:
                    continue
                count += 1

                image[x, y] = 60

                indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                for dx, dy in indices:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= height or ny < 0 or ny >= width or image[
                        nx, ny] == 60:
                        continue

                    if image[nx, ny] == 255:
                        to_it = (nx, ny)
                        queue.clear()
                        break
                    if visited.get(to_it) == None:
                        queue.append((nx, ny))
                visited[(x, y)] = True

            if to_it == None:
                break

            points = dfs(image, to_it[0], to_it[1], to_replace=255, replace_with=120)
            last_pt = points[len(points) - 1]

            points = dfs(image, last_pt[0], last_pt[1], to_replace=120, replace_with=60)
            if (len(points) > 20):
                contours.append(points)

            to_it = points[len(points) - 1]

    for x in range(height):
        for y in range(width):
            if visited.get((x, y)) == None:
                bfs(x, y)
    # cv2.imshow("After contours", image)

# 2D contours to 1D
#     points = []
#     for i in range(len(contours)):
#         cnt = contours[i]
#         for pt in cnt:
#             points.append((pt[0] / 2, pt[1] / 2))
#     return points
    return contours

def thresholdImage(img,thres,threshold):
    thres = thres.copy()
    row,col = thres.shape
    for i in range(row):
        for j in range(col):
            if(img[i,j]<threshold):
                thres[i,j] = 255
    return thres