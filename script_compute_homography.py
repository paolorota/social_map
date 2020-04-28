import cv2
import numpy as np


sample_image = '/data/social_map/behave01/0001.png'
sample_map = '/data/social_map/map_behave.png'
homo_filename = '/data/social_map/homo_behave.txt'

mouseX = 0
mouseY = 0
points_original = list()
points_map = list()

def draw_circle_original(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(img, str(len(points_original) + 1), (x, y), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 255, 255))
        mouseX, mouseY = x, y
        print('x = {}, y = {}'.format(mouseX, mouseY))
        points_original.append([x, y])

def draw_circle_map(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(map, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(map, str(len(points_map) + 1), (x, y), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 255, 255))
        mouseX, mouseY = x, y
        print('x = {}, y = {}'.format(mouseX, mouseY))
        points_map.append([x, y])


img = cv2.imread(sample_image)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle_original)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

## generating map
map = cv2.imread(sample_map)
# map = np.zeros_like(img, dtype=np.uint8)
dist = 20
width = map.shape[0]
height = map.shape[1]
for i in np.arange(0, width, dist):
    cv2.line(map, (0, i), (height, i), (255,255,255))
for i in np.arange(0, height, dist):
    cv2.line(map, (i, 0), (i, width), (255,255,255))

cv2.namedWindow('map')
cv2.setMouseCallback('map', draw_circle_map)
while(1):
    cv2.imshow('map', map)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

print('image points: {}'.format(len(points_original)))
print('map points: {}'.format(len(points_map)))

src = np.asarray(points_original)
dst = np.asarray(points_map)

h, status = cv2.findHomography(src, dst, method=cv2.RANSAC)
print(h)
cv2.destroyAllWindows()

# h = np.asarray(h)
warped = cv2.warpPerspective(img, h, dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
cv2.namedWindow('warped')
while(1):
    cv2.imshow('warped', warped)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

with open(homo_filename, 'w') as of:
    h.tofile(of, " ")
