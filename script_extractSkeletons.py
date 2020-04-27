import pyopenpose as op
import glob
from os.path import join
import cv2
import numpy as np
import tqdm
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


source_dir = '/data/social_map/behave01'
output_file = '/data/social_map/list.txt'
model_folder = "/home/prota/Desktop/openpose/models"

filelist = glob.glob(join(source_dir, '*.png'))
filelist.sort()

### OPENPOSE PARAMS
params = dict()
params["model_folder"] = model_folder
params["face"] = False
params["hand"] = False
params["num_gpu"] = 1
params["num_gpu_start"] = 0

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

with open(output_file, 'w') as of:
    for i, f in enumerate(tqdm.tqdm(filelist, desc='Files to check if skeleton is present:')):
        im = cv2.imread(f)
        im_size = (im.shape[1], im.shape[0])
        datum = op.Datum()
        datum.cvInputData = im
        opWrapper.emplaceAndPop([datum])

        skeletal_coordinates = np.around(np.array(datum.poseKeypoints).tolist(), 2).tolist()
        d = dict()
        try:
            d['file'] = f
            d['n_skeletons'] = len(skeletal_coordinates)
            # of.write('{} {} '.format(f, len(skeletal_coordinates)))
            pos = list()
            for ske in skeletal_coordinates:
                heels = list()
                lh = np.asarray(ske[21][:2], dtype=np.int32)
                if lh.any() != 0:
                    heels.append(lh)
                rh = np.asarray(ske[24][:2], dtype=np.int32)
                if rh.any() != 0:
                    heels.append(rh)
                av = [a.tolist() for a in heels]
                if len(av) > 0:
                    av = np.mean(av, axis=0, dtype=np.int32)
                    # im = cv2.circle(im, av, 5, (255, 0, 0))
                    # cv2.imshow('image', im)
                    # cv2.waitKey(0)
                    pos.append(av.tolist())
            d['skels'] = pos
        except TypeError:
            continue
        j = json.dumps(d)
        of.write(j + '\n')
        of.flush()

