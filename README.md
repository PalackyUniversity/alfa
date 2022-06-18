# Alfan
Program with graphical user interface for analyzing and evaluating field heights over time captured by LiDAR on a drone.  

## How it works

In `main-part1.py` program for each `las` file in `data` folder:
1. Loads 3D `las` file
2. Removes 3D points with bigger value than 99.9% quantile and lower than 0.1% quantile
3. Subtracts minimal `X`, `Y`, `Z` value and saves that values into `info.json`
4. Normalizes `X`, `Y` axes, then scales them by `WH_SCALE` factor
5. Normalizes `Z` axis, then scales it to 16bit image
6. Uses **dynamic averaging** - multiple 3D points at the same position in pixel grid are averaged, then 3D point count is visualized in `1_count.png` 

![Count](docs/1_count_compressed.png "Count")

7. Converts 3D data into 2D pixel grid, saves it into `1_original.png`

![Original](docs/1_original_compressed.png "Original")

8. Makes **median** of `1_original.png` that ignores zeroes when calculating median and then every black pixel replaces with median value => maximizes information from just few 3D points. Then saves it into `2_median.png`

![Median](docs/2_median_compressed.png "Median")

In `main-part2.py` then for last image in time series:

1. User rotates image using GUI, so that longer sides of blocks of field are vertical, program saves it into `3_rotated.png`

![Rotation](docs/3_rotated_compressed.png "Rotation")

2. User selects field for analysis from rotated image using GUI, program saves it into `4_cropped.png`

![Crop](docs/4_cropped_compressed.png "Crop")

3. Program automatically rotates and crops all other images

In `main-part3.py` program:
1. Sums all images into one image, saves it into `0_sum.png`

![Sum](docs/0_sum_compressed.png "Sum")

2. Sums summed image over vertical axis, applies **log** for better processing, saves it into `blocks_sum.png`

![Blocks sum](docs/blocks_sum.png "Blocks sum")

3. Detects starting and ending edges of blocks using first derivative, saves it into `blocks_derivative.png`

![Blocks derivative](docs/blocks_derivative.png "Blocks derivative")

4. Separates each block into separate image, saves example output in new folder - e.g. `block_1/0_block.png`

![Block](docs/0_block_compressed.png "Block")

5. Split block into upper and bottom part

For upper and bottom part program:
1. Sums block over vertical axis, applies **log** for better processing, e.g. saves it into `block_1/fit_up_sum.png`

![Fit sum](docs/fit_up_sum.png "Fit sum")

2. Detects starting and ending edges of sub-blocks using first derivative, e.g. saves it into `block_1/fit_up_derivative.png`

![Fit derivative](docs/fit_up_derivative.png "Fit derivative")

Then program puts upper and bottom edge detections together and:

1. Detects deformation

<!-- ![Block deformation](docs/0_block_compressed.png "Block deformation") -->

![Block deformation draw](docs/0_block_draw_compressed.png "Block deformation draw")

2. Removes deformation

<!-- ![Without deformation](docs/1_warped_compressed.png "Without deformation") -->

![Without deformation draw](docs/1_warped_draw_compressed.png "Without deformation draw")

3. Applies local maxima filter (with kernel size 1/10 sub-block width)

![Maxima](docs/1_warped_max_compressed.png "Maxima")

Finally, for each sub-block program:

1. Subtracts from it the same sub-block from first las file (the earliest record in time series when nothing has grown yet = terrain)

![Terrain](docs/terrain_compressed.png "Terrain")

2. Computes:
   - Mean
   - Standard deviation
   - Median
   - Minimum
   - Maximum
   - Quantiles - 1%, 5%, 25%, 75%, 95%, 99%
   - Relative growth rate using formula:
<p align="center">
 <img src="docs/rgr.svg">
</p>

3. Some keys saves into `statistics.csv`

4. Key enumerating image is saved into `4_cropped_draw.png`

![Analysis](docs/4_cropped_draw_compressed.jpg "Analysis")

## Structure

### Folders
- `data` - input folder for las files
  - `210420_065626.las` - datetime in filename `YYMMDD_HHMMSS *.las`, e.g.:
  - `210504_064914.las`
  - ...
- `results` - output folder
  - `210420_065626` - match with filename from data without extension 
    - `1_count.png` 
    - `1_original.png`
    - `2_median.png`
    - ...
  - `210504_064914`
    - ...
    
### Files

- `main.c` - optimized C script for converting 3D las data to 2D image and computing median 
- `main-part1.py` - Python wrapper for C script
- `main-part2.py` - User GUI for precise field rotation and field selection
- `main-part3.py` - Python script for analysis
- `init.py` - constants + common things

## Install on Debian based Linux:

### Install Python3 and gcc

```bash
sudo apt update
sudo apt install python3 python3-pip build-essential
```

### Install Python packages
```bash
python3 -m pip install -r requirements.txt
```

### Compile main.c

```bash
gcc -shared -o main.so main.c
```

## Run
1. Paste/move your las files into data folder (match folder and file structure)
2. Run `python3 main-part1.py` to generate images
3. Run `python3 main-part2.py`
   1. Rotate image using arrows, so field blocks lines are vertical
   2. Press *enter* or key *q*
   3. Select upper left corner using *arrows* to crop
   4. Press *enter* or key *q*
   5. Select bottom right corner using *arrows* to crop
   6. Press *enter* or key *q*
4. Run `python3 main-part3.py` to create analysis
5. Results are in `results` folder

## Tested on:
- MacOS 12 Python 3.8
- Ubuntu 22.04 Python 3.10


## TODO
- [ ] Optimize for multicore processors
- [ ] Optimize for low RAM computers
- [ ] Object Oriented Programming?
- [ ] Python classes for dicts with standardized keys¨
- [ ] More robust sub-field search? 
  - Find missing peaks by fitting periodic function?
  - Auto crop black padding at the ends?