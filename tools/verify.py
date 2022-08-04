# diff = images[2].astype(int) - images[1].astype(int)
# control_diff_mene = diff.copy()
# control_diff_vice = diff.copy()
#
# print(control_diff_vice.min(), control_diff_mene.min(), control_diff_vice.max(), control_diff_mene.max())
#
# control_diff_mene[control_diff_mene < 0] = 0
# print(control_diff_vice.min(), control_diff_mene.min(), control_diff_vice.max(), control_diff_mene.max())
#
# control_diff_vice[control_diff_vice > 0] = 0
# control_diff_vice = np.abs(control_diff_vice)
#
# print(control_diff_vice.min(), control_diff_mene.min(), control_diff_vice.max(), control_diff_mene.max())
# print(control_diff_mene.dtype, control_diff_vice.dtype)
# cv2.imwrite("test_mene.png", np.uint16(control_diff_mene*100))
# cv2.imwrite("test_vice.png", np.uint16(control_diff_vice*100))
#
# exit()
# print(files[0:3])
# control_diff = cv2.absdiff(images[2], images[1])
# print(np.max(control_diff) / Z_MAX * (info["z_max"] - info["z_min"]) * 100)
# print(np.max(control_diff))
# print(np.mean(control_diff) / Z_MAX * (info["z_max"] - info["z_min"]) * 100)
# #control_diff[control_diff < 1_000] = 0
# cv2.imwrite("test.png", cv2.resize(control_diff*50, None, fx=1/4, fy=1/4))
# cv2.waitKey(0)
