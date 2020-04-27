# social_map

This software produces a _social heatmap_ according to the density of people within a certain area.

## Instructions
Tested on [VIRAT dataset](https://viratdata.org/), where videos has been converted using _ffmpef_:
```bash
ffmpeg -i VIRAT_S_000001.mp4 -r 1/1 -qscale:v 2 video01/$filename%04d.png
```  

## Credits to
[Open CV](https://opencv.org/)

[Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)