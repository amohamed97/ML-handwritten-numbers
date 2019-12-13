import gzip
import sys
import random
import numpy
import math
from scipy.spatial import distance
import timeit
from random import randrange
import copy

def print_image(img):
    for i in range(28):
        for j in range(28):
            if img[(i*28)+j] > 0:
                sys.stdout.write("\033[1;31m")
                sys.stdout.write("x")
                sys.stdout.write("\033[0;0m")
            else:
                sys.stdout.write("0")
        sys.stdout.write("\n")

def euc_distance(img1, img2):
    d = len(img1)
    distance = 0
    for i in range(d):
        distance += (img1[i]-img2[i])**2
    return math.sqrt(distance)

## Reading Training Set
f = gzip.open("training-imgs.gz")
file_content = f.read()
set_count = 10000
data_set = []
for i in range(0,set_count):
    data_set.append(list(file_content[(i*784)+16:((i+1)*784)+16]))
##
## Picking centroids
k = 10
centroids = []
centroids_numpy = []
first_centroid = randrange(set_count-20)
for i in range(k):
    centroids.append([copy.deepcopy(data_set[first_centroid+i]),[]])
    centroids_numpy.append(numpy.array(data_set[first_centroid+i]))
##
## Classifying the whole data set
for itr in range(50):
    old_centroids = []
    print("Iteration #" + str(itr))
    for i in range(k):
        old_centroids.append(len(centroids[i][1]))
        centroids[i][1] = []
    for i in range(set_count):
        min_index = 0
        min_distance = 1000000000
        current_image = numpy.array(data_set[i])
        for j in range(k):
            dst = distance.euclidean(current_image, centroids_numpy[j])
            if dst < min_distance:
                min_index = j
                min_distance = dst
        centroids[min_index][1].append(data_set[i])
    not_yet = 0
    for i in range(k):
        if len(centroids[i][1]) != old_centroids[i]:
            not_yet = 1
            break
    if not_yet == 0:
        print("Done training and it's perfect")
        break
    for i in range(k):
        centroids[i][0] = list(numpy.mean(centroids[i][1], axis=0))
        centroids_numpy[i] = numpy.array(centroids[i][0])
##



## Reading Training Set
f = gzip.open("test_imgs.gz")
file_content = f.read()
test_count = 10000
test_set = []
for i in range(0,test_count):
    test_set.append(list(file_content[(i*784)+16:((i+1)*784)+16]))

f = gzip.open("test_labels.gz")
file_content = f.read()
test_labels_set = []
for i in range(0,test_count):
    test_labels_set.append(file_content[8+i])
##

predicted_set = []

for i in range(0, test_count):
        min_index = 0
        min_distance = 1000000000
        current_image = numpy.array(test_set[i])
        for j in range(k):
            dst = distance.euclidean(current_image, centroids_numpy[j])
            if dst < min_distance:
                min_index = j
                min_distance = dst
        predicted_set.append(min_index)

right = 0
wrong = 0
for i in range(test_count):
    if test_labels_set[i] == predicted_set[i]:
        right += 1
    else:
        wrong += 1

print(str(right))
print(str(wrong))
## Visualizing an image
# for i in range(28):
#     for j in range(28):
#         if data_set[50000][(i*28)+j] > 0:
#             sys.stdout.write("\033[1;31m")
#             sys.stdout.write("x")
#             sys.stdout.write("\033[0;0m")
#         else:
#             sys.stdout.write("0")
#     sys.stdout.write("\n")

