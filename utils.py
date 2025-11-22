import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import tqdm

class Backend(Enum):
    CV2 = "cv2"
    PLT = "plt"

CV2_MAX_IMAGE_HEIGHT = 1080
SHOW_IMAGE_BACKEND = Backend.PLT
DRAW_MATCH_THICKNESS = 2


def warp_perspective(image, rot_deg=0.0, shear_x=0.0, shear_y=0.0, scale=1.0, tx=0.0, ty=0.0, normalize_shear=1000.0):
    """
    Apply a homography warp to the input image using intuitive parameters:
    - rotation in degrees (about origin)
    - shear_x, shear_y are treated as shear terms and normalized (divide by `normalize`)
    - uniform scale
    - tx, ty translations (in same units as image pixels after warp)
    - persp is optional perspective term placed in H[2,0]; if None it's derived small from skew
    """
    H = __make_homography(rot_deg, shear_x, shear_y, scale, tx, ty, normalize_shear)
    warped_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    return warped_image

def __make_homography(rot_deg=0.0, shear_x=0.0, shear_y=0.0, scale=1.0, tx=0.0, ty=0.0, normalize_shear=1000.0):
    """
    Build a homography from intuitive parameters:
    - rotation in degrees (about origin)
    - shear_x, shear_y are treated as shear terms and normalized (divide by `normalize`)
    - uniform scale
    - tx, ty translations (in same units as image pixels after warp)
    - persp is optional perspective term placed in H[2,0]; if None it's derived small from skew
    """
    ang = np.deg2rad(rot_deg)
    c, s = np.cos(ang), np.sin(ang)

    # normalized shear so user can pass large pixel-like values and get reasonable homography
    sx = float(shear_x) / float(normalize_shear)
    sy = float(shear_y) / float(normalize_shear)

    # small perspective term (kept tiny)
    p = (shear_x + shear_y) / (float(normalize_shear) * 1e3)

    # rotation + scale
    R = np.array([[scale * c, -scale * s, tx],
                  [scale * s,  scale * c, ty],
                  [0.0,        0.0,       1.0]])

    # shear (skew)
    Sh = np.array([[1.0, sx,  0.0],
                   [sy,  1.0, 0.0],
                   [0.0, 0.0, 1.0]])

    # perspective
    P = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [p,   0.0, 1.0]])

    # compose: first shear, then rotate+scale, then small perspective
    H = P @ R @ Sh
    return H


def show_inlier_matches(image1, keypoints1, image2, keypoints2, matches, inlier_mask):

    inlier_mask_list = inlier_mask.astype(int).tolist()
    outlier_mask_list = (~inlier_mask).astype(int).tolist()

    out_image_inliers_1 = cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches, None,
        matchColor=(0,0,255), # draw in red color for outliers
        matchesThickness=DRAW_MATCH_THICKNESS,
        matchesMask=outlier_mask_list, # draw only outliers
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches, out_image_inliers_1,
        matchColor=(0,255,0), # draw in green color for inliers
        matchesThickness=DRAW_MATCH_THICKNESS,
        matchesMask=inlier_mask_list, # draw only inliers
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS | cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    )

    out_image_inliers_2 = cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches, None,
        matchColor=(0,255,0), # draw in green color for inliers
        singlePointColor=(0,0,255),
        matchesThickness=DRAW_MATCH_THICKNESS,
        matchesMask=inlier_mask_list, # draw only inliers
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    n_inliers = np.sum(inlier_mask)
    n_outliers = len(inlier_mask) - n_inliers
    show_images(
        [out_image_inliers_1, out_image_inliers_2],
        [f"{n_inliers} Inliers (green) and {n_outliers} Outliers (red)", "Inliers (green) only"])

def non_maxima_suppression(keypoints, image_shape, windowsize=10):
    kernel = np.ones((windowsize, windowsize), dtype=np.uint8)
    response_image = np.zeros(image_shape[:2], dtype=np.float32)
    for kp in keypoints:
        response_image[int(kp.pt[1]), int(kp.pt[0])] = kp.response
    dilated = cv2.dilate(response_image, kernel)
    nms_keypoints = []
    for kp in keypoints:
        if response_image[int(kp.pt[1]), int(kp.pt[0])] == dilated[int(kp.pt[1]), int(kp.pt[0])]:
            nms_keypoints.append(kp)
    return nms_keypoints

def show_image(image, title="", *args, **kwargs):
    if SHOW_IMAGE_BACKEND == Backend.PLT:
        __show_image_plt(image, title, *args, **kwargs)
    else:
        __show_image_cv2(image, title, *args, **kwargs)

def show_images(images, titles, *args, **kwargs):
    if SHOW_IMAGE_BACKEND == Backend.PLT:
        __show_images_plt(images, titles, *args, **kwargs)
    else:
        __show_images_cv2(images, titles, *args, **kwargs)

def __show_image_cv2(image, title="", *args, **kwargs):
    if CV2_MAX_IMAGE_HEIGHT is not None:
        height, width = image.shape[:2]
        if height > CV2_MAX_IMAGE_HEIGHT:
            scale = CV2_MAX_IMAGE_HEIGHT / height
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def __show_images_cv2(images, titles, *args, **kwargs):
    # stick images horizontally, padding with black if needed
    # Moreover, take care of channels if of different types (grayscale vs color)
    for img_idx in range(len(images)):
        if CV2_MAX_IMAGE_HEIGHT is not None:
            height, width = images[img_idx].shape[:2]
            if height > CV2_MAX_IMAGE_HEIGHT:
                scale = CV2_MAX_IMAGE_HEIGHT / height
                images[img_idx] = cv2.resize(images[img_idx], (0, 0), fx=scale, fy=scale)

    max_height = max(image.shape[0] for image in images)
    n_channels = max(image.shape[2] if len(image.shape) == 3 else 1 for image in images)
    for img_idx in range(len(images)):
        image = images[img_idx]
        height, width = image.shape[:2]
        if height < max_height:
            pad_height = max_height - height
            image = cv2.copyMakeBorder(image, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if (len(image.shape) == 2 and n_channels > 1) or (len(image.shape) == 3 and image.shape[2] != n_channels == 3):
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        images[img_idx] = image

    output_image = cv2.hconcat(images)
    cv2.imshow(" | ".join(titles), output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def __show_image_plt(image, title="", *args, **kwargs):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), *args, **kwargs)
    plt.title(title)
    plt.axis('off')
    plt.show()

def __show_images_plt(images, titles, *args, **kwargs):
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB), *args, **kwargs)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

try:
    from scipy.cluster.vq import _asarray_validated, _kpoints, _kmeans, check_random_state
except ImportError:
    pass

def kmeans(obs, k_or_guess, iter=20, thresh=1e-5, check_finite=True,
           *, seed=None):
    try:
        obs = _asarray_validated(obs, check_finite=check_finite)
        if iter < 1:
            raise ValueError("iter must be at least 1, got %s" % iter)

        # Determine whether a count (scalar) or an initial guess (array) was passed.
        if not np.isscalar(k_or_guess):
            guess = _asarray_validated(k_or_guess, check_finite=check_finite)
            if guess.size < 1:
                raise ValueError("Asked for 0 clusters. Initial book was %s" %
                                guess)
            return _kmeans(obs, guess, thresh=thresh)

        # k_or_guess is a scalar, now verify that it's an integer
        k = int(k_or_guess)
        if k != k_or_guess:
            raise ValueError("If k_or_guess is a scalar, it must be an integer.")
        if k < 1:
            raise ValueError("Asked for %d clusters." % k)

        rng = check_random_state(seed)

        # initialize best distance value to a large value
        best_dist = np.inf
        for i in tqdm.tqdm(range(iter)):
            # the initial code book is randomly selected from observations
            guess = _kpoints(obs, k, rng)
            book, dist = _kmeans(obs, guess, thresh=thresh)
            if dist < best_dist:
                best_book = book
                best_dist = dist
        return best_book, best_dist
    except Exception as e:
        from scipy.cluster.vq import kmeans as scipy_kmeans
        return scipy_kmeans(obs, k_or_guess, iter=iter, thresh=thresh, check_finite=check_finite, seed=seed)
