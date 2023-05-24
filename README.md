# OneShotSlam
implementation of basic monocular SLAM

#Libraries Used

Opencv for feature extraction

Open3d for 3-D display

Numpy  for math

# Usage
python ./app/main.py


# Description
In imgs folder:
Gt3Dpoint.png is ground truth points in 3D world.

ReferenceResult.png is perspective of origin to estimated points 

Result2.png is estimated points that with another view.

Result2.png is estimated points that with another view.

gtpoint.png iis ground truth points in 2D world.

img1Point2D.png is extracted 2D points for reference img1.

img2Point2D.png is extracted 2D points for img2.

img1img2Extract3dPoints.png is est. 3D points that extracted by img1 and img2.

img2img3Extract3dPoints.png is est. 3D points that extracted by img2 and img3

trajectory_XZ.png is est. trajectory on X and Z axis.

trajectory_XYZ.png is est. trajectory that every combination for all axis.

In data folder:
EstPoint3d.npy filw is est. 3D point that saved as numpy file.
