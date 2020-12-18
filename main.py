# Автор Евгений Казаченко гр.821701
# Вариант 8
# Лабораторная работа №1
import math
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread


# Functions #

# Print information about this neural network
def printInformation(block_height, block_width, height, width, input_layer_size, hidden_layer_size, number_of_blocks):
    print("Вывод данных")
    print(f'Количество блоков длины: {block_height}')
    print(f'Количестов блоков ширины: {block_width}')
    print(f'Высота изображения: {height}')
    print(f'Ширина изображения: {width}')
    print(f'Входной слой: {input_layer_size}')
    print(f'Второй слой: {hidden_layer_size}')
    print(f'Количество блоков: {number_of_blocks}')


# Normalize_matrix counts adaptive learning step
def adaptive_learning_step(matrix):
    result = 1.0 / (np.matmul(matrix, np.transpose(matrix) * 10))
    return result


# Normalize_matrix normalizes given matrix
# Author by Maria Zhirko and Xolypko Alexsandr group.821701 #
def normalize_matrix(matrix):
    for i_f in range(len(matrix[0])):
        s = 0
        for j_f in range(len(matrix)):
            s += matrix[j_f][i_f] * matrix[j_f][i_f]
        s = math.sqrt(s)
        for j_f in range(len(matrix)):
            matrix[j_f][i_f] = matrix[j_f][i_f] / s


# vector deleted
def to_blocks(array):
    blocks = []
    for i in range(height // block_height):
        for j in range(width // block_width):
            block = []
            for y in range(block_height):
                for x in range(block_width):
                        block.append(array[i * block_height + y, j * block_width + x, 0])
                        block.append(array[i * block_height + y, j * block_width + x, 1])
                        block.append(array[i * block_height + y, j * block_width + x, 2])
            blocks.append(block)
    return np.array(blocks)


# Show picture
def show(picture):
    picture = 1.0 * (picture + 1) / 2
    plt.imshow(picture)
    plt.show()


# restore image
def to_array(blocks):
    restore_image = []
    blocks_in_line = width // block_width
    for i in range(height // block_height):
        for y in range(block_height):
            line = []
            for j in range(blocks_in_line):
                for x in range(block_width):
                    pixel = [blocks[i * blocks_in_line + j, (y * block_width * 3) + (x * 3) + 0],
                             blocks[i * blocks_in_line + j, (y * block_width * 3) + (x * 3) + 1],
                             blocks[i * blocks_in_line + j, (y * block_width * 3) + (x * 3) + 2]]
                    line.append(pixel)
            restore_image.append(line)
    return np.array(restore_image)


# Main Program #

# Load image
image = imread("Image/jotaro.png")
image = (2.0 * image / 1.0) - 1.0
array = np.array(image)

height = np.size(array, 0)
width = np.size(array, 1)

# Enter n and m
block_height = int(input())
block_width = int(input())

# Enter error
error_max = int(input())

# Enter hidden_layer_size
hidden_layer_size = int(input())  # p

# Count the size
color = 3  # S(R,G,B)
number_of_blocks = int((height / block_height) * (width / block_width))  # L
input_layer_size = block_height * block_width * color  # N

# hidden_layer_size =  input_layer_size // 3

# Void information about neural_work
printInformation(block_height, block_width, height, width, input_layer_size, hidden_layer_size, number_of_blocks)

blocks = to_blocks(array).reshape(number_of_blocks, 1, input_layer_size)

# w1 = np.random.rand(input_layer_size, hidden_layer_size) * 2 - 1
w1 = []

# uniform random distribution from (-1;1)
for i in range(input_layer_size):
    tmpList = []
    w1.append(tmpList)
    for j in range(hidden_layer_size):
        weight = random.uniform(-1, 1)
        tmpList.append(weight)
blocks = np.array(blocks)

w1 = np.array(w1)  # scales_1 = N*p
temp = np.copy(w1)
w2 = temp.transpose()  # scales_2 = w1^t

z = (input_layer_size * number_of_blocks) / ((input_layer_size + number_of_blocks) * hidden_layer_size + 2)
print(f'Z = {z}')

# alpha_one = 0.0005
# alpha_two = 0.0005

error_current = error_max + 1
counter = 0

# Training neural network
while error_current > error_max:
    error_current = 0
    counter += 1
    for i in blocks:
        y = np.matmul(i, w1)
        x1 = np.matmul(y, w2)
        dx = x1 - i

        alpha_one = adaptive_learning_step(i)
        w1 -= alpha_one * np.matmul(np.matmul(i.transpose(), dx), w2.transpose())  # learn first

        alpha_two = adaptive_learning_step(y)
        w2 -= alpha_two * np.matmul(y.transpose(), dx)  # learn second

        normalize_matrix(w1)
        normalize_matrix(w2)

    for i in blocks:
        dx = np.matmul(np.matmul(i, w1), w2) - i
        error = (dx * dx).sum()  # root mean square error
        error_current += error
    print('Epoch:', counter)
    print('Error:', error_current)

print("Done")

# author: ARtoriouSs
# link: https://github.com/ARtoriouSs
result_work = []
for block in blocks:
    result_work.append(block.dot(w1).dot(w2))
result_work = np.array(result_work)

# author: Yauheni Kazachenka gr.821701
print("Original picture:")
show(array)
print("Neural network picture")
show(to_array(result_work.reshape(number_of_blocks, input_layer_size)))
###############################################################################
# numpy.matmul - матричное произведение двух массивов == @
# numpy.dot - скалярное произведение двух массивов
