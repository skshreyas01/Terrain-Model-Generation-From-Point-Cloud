import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

flatter = 1.0 / 298.257224
fuse_file = 'final_project_point_cloud.fuse'
ecef_output = 'ecef_output'
N = 100.0


def read_file(fileName):
    with open(fileName, 'r') as file:
        file_reader = csv.reader(file, delimiter=' ')
        lines = list(file_reader)
    return [list(map(float, line)) for line in lines]


def write_file(lines, fileName):
    with open(fileName, 'w') as file:
        for line in lines:
            file.write('%s\n' % ' '.join(line))

def convert_to_matrix(lines):
    X, Y, Z, I = np.array(lines).T
    min_x, max_x, min_y, max_y = min(X), max(X), min(Y), max(Y)
    diff = (max_x - min_x) / N
    Y_diff = int((max_y - min_y) / diff)

    matrix = [[[] for _ in xrange(int(Y_diff))] for _ in xrange(int(N))]

    for x, y, z, i in lines:
        xi = int((x - min_x) / diff)
        yi = int((y - min_y) / diff)
        matrix[xi - 1][yi - 1].append((x, y, z, i))

    line_list = []
    lines_list = []
    X_len, Y_len = len(matrix), len(matrix[0])
    for i in xrange(X_len):
        for j in xrange(Y_len):
            line_list.append(min(matrix[i][j], key=lambda n: n[2])[2] if matrix[i][j] else 222.34)
        lines_list.append(line_list)
        line_list = []
    gaussian_matrix = np.array(lines_list)
    image_block = np.ones((6, 6), np.uint8)

    line_list = []
    lines_list = []
    overlay = cv2.erode(gaussian_matrix, image_block, iterations=1)
    for i in xrange(X_len):
        for j in xrange(Y_len):
            line_list.append(overlay[i][j] if overlay[i][j] != 223 else 223)
        lines_list.append(line_list)
        line_list = []
    overlay = np.array(lines_list)
    overlay = cv2.dilate(overlay, image_block, iterations=3)


    line_list = []
    lines_list = []
    for i in xrange(X_len):
        for j in xrange(Y_len):
            line_list.append(1 if overlay[i][j] >= 224.5 else 0)
        lines_list.append(line_list)
        line_list = []
    overlay = np.array(lines_list)

    image_block = np.ones((10, 10), np.uint8)
    gaussian_matrix = cv2.erode(gaussian_matrix, image_block, iterations=1)

    return gaussian_matrix, overlay


def ecef(line):
    lat, lon, alt, intensity = map(float, line)

    cosLat = math.cos(lat * math.pi / 180.0)
    sinLat = math.sin(lat * math.pi / 180.0)
    cosLon = math.cos(lon * math.pi / 180.0)
    sinLon = math.sin(lon * math.pi / 180.0)

    C = 1.0 / math.sqrt(cosLat * cosLat + (1 - flatter) * (1 - flatter) * sinLat * sinLat)
    x = (6378137.0 * C + 0.0) * cosLat * cosLon
    y = (6378137.0 * C + 0.0) * cosLat * sinLon
    z = alt

    return map(str, [x, y, z, intensity])


if __name__ == '__main__':
    fuse_reader = map(ecef, read_file(fuse_file))
    write_file(fuse_reader, ecef_output)
    if (os.path.exists(ecef_output)):
        lines = read_file(ecef_output)
        gaussian_matrix, mask = convert_to_matrix(lines)

        gaussian_matrix = np.array(map(lambda n: map(float, n), gaussian_matrix))

        gaussian_blur = cv2.GaussianBlur(gaussian_matrix, (9, 9), 0)
        mean_val = gaussian_matrix[mask == 1].mean()

        line_list = []
        lines_list = []
        for i in xrange(len(gaussian_matrix)):
            for j in xrange(len(gaussian_matrix[0])):
                line_list.append(mean_val if mask[i][j] else gaussian_blur[i][j])
            lines_list.append(line_list)
            line_list = []
        image = np.array(lines_list)
        image = cv2.GaussianBlur(image, (9, 9), 0)
        plt.imshow(image)
        plt.colorbar()
        plt.show()