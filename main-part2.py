from init import *
import numpy as np
import imutils
import cv2

images = [cv2.imread(file, cv2.CV_16U) for file in sorted(glob.glob(f"{PATH_RESULTS}/*/{IMAGE_MEDIAN}{EXTENSION}"))]

assert len(images) != 0, "Missing images from part1!"

# Image rotation - init parameters
rotation_angle: float = 0
step: float = 1
recompute: bool = True

# Image rotation by user - GUI
while True:
    if recompute:
        recompute = False
        rotated_image = imutils.rotate_bound(images[-1], rotation_angle)

        cv2.imshow(f"window", rotated_image)
    cv2.setWindowTitle("window", f"Rotation - use arrows - step {step:.1f} deg. - angle {rotation_angle:.1f} deg.")

    key = cv2.waitKey(0)

    # Update rotation angle
    if key == 2:  # Arrow left
        rotation_angle -= step
        recompute = True

    elif key == 3:  # Arrow right
        rotation_angle += step
        recompute = True

    # Increase decrease step size
    elif key == ord("w") and step + GUI_STEP_ANGLE <= 180:  # Key "w"
        step += GUI_STEP_ANGLE

    elif key == ord("s") and step - GUI_STEP_ANGLE >= 0:  # Key "s"
        step -= GUI_STEP_ANGLE

    # Finish rotation
    elif key == 13 or key == ord("q"):  # Key Enter or key "q"
        break

# Crop image - init parameters
point_lu = np.array(rotated_image.shape[:2]) // GUI_CROP_INIT
point_rb = np.array(rotated_image.shape[:2]) // GUI_CROP_INIT * (GUI_CROP_INIT - 1)
step = min(rotated_image.shape[:2]) // GUI_CROP_STEP

cropped_image = rotated_image.copy()

for point in [point_lu, point_rb]:
    recompute = True

    while True:
        if recompute:
            cv2.imshow(f"window", cropped_image[point_lu[0]:point_rb[0], point_lu[1]:point_rb[1]])
            recompute = False

        cv2.setWindowTitle("window", f"Crop image - use arrows - step {step:.1f} px")

        key = cv2.waitKey(0)

        if key == 2:  # Arrow left
            point[1] -= step
            recompute = True

        elif key == 3:  # Arrow right
            point[1] += step
            recompute = True

        elif key == 0:  # Arrow up
            point[0] -= step
            recompute = True

        elif key == 1:  # Arrow down
            point[0] += step
            recompute = True

        elif key == ord("w") and step + GUI_STEP_PX <= max(rotated_image.shape) / 2:  # Key "w"
            step += GUI_STEP_PX

        elif key == ord("s") and step - GUI_STEP_PX >= 0:  # Key "s"
            step -= GUI_STEP_PX

        elif key == 13 or key == ord("q"):  # Key Enter or key "q"
            break

# Save images
for n, image in enumerate(images):
    rotated_image = imutils.rotate_bound(image, rotation_angle)
    cropped_image = rotated_image[point_lu[0]: point_rb[0], point_lu[1]: point_rb[1]]

    cv2.imwrite(f"{PATH_RESULTS}/{files[n]}/{IMAGE_ROTATED}{EXTENSION}", rotated_image)
    cv2.imwrite(f"{PATH_RESULTS}/{files[n]}/{IMAGE_CROPPED}{EXTENSION}", cropped_image)
