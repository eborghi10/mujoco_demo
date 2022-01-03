'''
Python script that loads a MuJoCo simulation and shows two RGB-D cameras with
pincushion/barrel distortion.

python3 camera_rendering_example.py

MUJOCO_HOME=/home/$USER/.mujoco/mujoco210
MJLIB_PATH=$MUJOCO_HOME/bin/libmujoco210.so
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_HOME/bin:/usr/lib/nvidia-000:/usr/lib/nvidia
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so.2.1
'''
from mujoco_py import MjSim, MjViewer, MjRenderContextOffscreen

from dt_apriltags import Detector
import mujoco_py
import numpy as np
import cv2
import time


# Initialize camera parameters
width, height = 1280, 720
dx, dy = 0.1, 0.1
k = -400e-8
y, x = np.mgrid[0:height:1, 0:width:1]
x = x.astype(np.float32) - width / 2 - dx
y = y.astype(np.float32) - height / 2 - dy
theta = np.arctan2(y, x)
r = (x * x + y * y) ** 0.5
rd = r * (1 - k * r * r)
map_x = rd * np.cos(theta) + width / 2 + dx
map_y = rd * np.sin(theta) + height / 2 + dy

# https://github.com/duckietown/lib-dt-apriltags
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       debug=0)

# Random values from here: https://github.com/duckietown/lib-dt-apriltags/blob/master/test/test.py#L41
intrinsics = (336.7755634193813, 336.02729840829176, 333.3575643300718, 212.77376312080065)


def add_distortion(undistorted_image):
    return cv2.remap(undistorted_image, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def render_opencv(height=height, width=width):
    # Two cameras exist in simulation
    for camera_index in [0, 1]:
        vieweroff.render(width, height, camera_id=camera_index)
        rgb, depth = vieweroff.read_pixels(width, height, depth=True, segmentation=False)
        bgr = np.flipud(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        distorted_bgr = add_distortion(bgr)
        cv2.imshow(f'camera{camera_index} color', distorted_bgr)

        # Detecting camera pose with end-effector camera
        if camera_index == 1:
            # Grayscale image
            distorted_bw = cv2.cvtColor(distorted_bgr, cv2.COLOR_BGR2GRAY)
            detections = at_detector.detect(
                distorted_bw, estimate_tag_pose=True, camera_params=intrinsics, tag_size=0.17)
            print("--------------------------")
            print(f"Tag Pose from camera: {detections[0].pose_t}, {detections[0].pose_R}")

        cv2.normalize(np.flipud(depth), depth, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow(f'camera{camera_index} depth', add_distortion(depth))

    cv2.waitKey(1)


model = mujoco_py.load_model_from_path("my-model-rrr.xml")

''' ATTENTION: if you choose to use Mujoco's default viewer, you can't see the rendering of the cameras!'''

sim = MjSim(model)
vieweroff = MjRenderContextOffscreen(sim, 0)
viewer = MjViewer(sim)

# controller and simulation params
t = 0
qpos_ref = np.array([-2, -1, 2])
qvel_ref = np.array([0, 0, 0])
kp = 1000
kv = 500

sim.model.opt.gravity[:] = np.array([0, 0, 0])  # just to make simulation easier :)

t_ini = time.time()

try:
    while True:
        # robot controller
        qpos_cur = sim.data.qpos
        qvel_cur = sim.data.qvel
        qpos_error = qpos_ref - qpos_cur
        qvel_error = qvel_ref - qvel_cur
        ctrl = qpos_error*kp + qvel_error*kv
        sim.step()
        viewer.render()
        render_opencv()

        continue

        t = t + 1
        sim.data.ctrl[:] = ctrl

        if t > 1000:
            break

except KeyboardInterrupt:
    print("Exit")

t_total = time.time() - t_ini

FPS = t / t_total

print(FPS, "fps")

cv2.destroyAllWindows()
