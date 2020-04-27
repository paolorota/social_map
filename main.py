import numpy as np
import pyopenpose as po
import cv2
import json
import time
from scipy.spatial.distance import cdist

list_file = '/data/social_map/list.txt'
homography_file = '/data/social_map/homo_behave.txt'
map_file = '/data/social_map/map_behave.png'
fileout = '/data/social_map/fileout.avi'
threshold = 50

def read_list(filename):
    with open(filename, 'r') as fin:
        filelist = fin.read().splitlines()
    framelist = list()
    for i in filelist:
        tmp = i.split(' ')
        frame = {
            'frame': tmp[0],
            'skeletons': int(tmp[1])
        }
        framelist.append(frame)
    return framelist


def read_jsonList(filename):
    with open(filename, 'r') as fin:
        data = fin.read().splitlines()
    dlist = list()
    for i in data:
        d = json.loads(i)
        dlist.append(d)
    return dlist


def readHomography(filename):
    with open(filename, 'r') as fin:
        filelist = fin.read().splitlines()
    h = np.zeros((0, 3), np.float)
    for i in filelist:
        tmp = i.split(',')
        tmp = np.array(tmp).astype(np.float).reshape((1, 3))
        h = np.concatenate((h, tmp), axis=0)
    return np.transpose(h)

def readBehaveHomo(filename):
    with open(filename, 'r') as of:
        r = np.fromfile(of, sep=" ")
    r = np.reshape(r, (3, 3))
    return r

def find_distances(points):
    dist = np.ones((len(points), len(points)), dtype=np.float32) * threshold * 2
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            xa = np.reshape(points[i], (1, 2))
            xb = np.reshape(points[j], (1, 2))
            dist[i, j] = cdist(xa, xb, metric='euclidean')
            dist[j, i] = dist[i, j]
    return dist

framelist = read_jsonList(list_file)
homography = readBehaveHomo(homography_file)

map = cv2.imread(map_file)
map_resized = cv2.resize(map, (640, 480))
cv2.namedWindow('image')
cv2.namedWindow('map')
vout = cv2.VideoWriter(fileout, fourcc=cv2.VideoWriter_fourcc('m','p','4','v'), fps=5, frameSize=(640, 480*2))
colorOK = (0, 255, 255)
colorNO = (0, 0, 255)
for i, frame in enumerate(framelist):
    image = cv2.imread(frame['file'])
    ellipses = np.zeros_like(map)
    # sz = list(image.shape[:2])
    dists = find_distances(frame['skels'])
    for j, points in enumerate(frame['skels']):
        pts = np.asarray(points, dtype=np.float32)
        pts = np.expand_dims(pts, axis=0)
        pts = np.expand_dims(pts, axis=0)
        dst1 = cv2.perspectiveTransform(pts, homography)
        dst = tuple(np.squeeze(dst1).astype(np.int).tolist())
        tmpc = colorOK
        if np.any(dists[j, :] < threshold * 2):
            tmpc = colorNO
        ellipses = cv2.circle(ellipses, dst, threshold, tmpc, thickness=-1)
    i_homo = np.linalg.inv(homography)
    ellipses = cv2.resize(ellipses, (640, 480))
    warped_ellipses = cv2.warpPerspective(ellipses, i_homo, dsize=(image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    image_ellipses = np.clip(image + warped_ellipses, 0, 255)
    map_ellipses = np.clip(map_resized + ellipses, 0, 255)
    cv2.imshow('image', image)
    cv2.imshow('map', map_ellipses)
    k = cv2.waitKey(20) & 0xFF
    # cv2.waitKey(0)
    if k == 27:
        break
    time.sleep(0.1)
    vout.write(np.concatenate((image, map_resized + ellipses), axis=0))

vout.release()
    # break