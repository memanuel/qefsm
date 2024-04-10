import numpy as np
import cv2

# This code was adapted from
# https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

# **************************************************************************************************
# Module constants
MAX_FEATURES = 4096
GOOD_MATCH_FRAC = 0.25
dtype = np.float64

# **************************************************************************************************
def align_image(img_aln: np.ndarray, img_ref: np.ndarray, img_tif: np.ndarray):
    """
    Align an OpenCV image to a reference image
    INPUTS:
        img_aln:    The image to align
        img_ref:    The reference image
        img_tif:    The original TIF image to be registered so it aligns with img_ref
    RETURNS:
        im1_reg:    A registered version of image 1, now aligned to the reference image
    """
 
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(img_aln, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img_ref, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Remove low quality matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_FRAC)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    n_match = len(matches)
    points1 = np.zeros((n_match, 2), dtype=dtype)
    points2 = np.zeros((n_match, 2), dtype=dtype)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography on the TIF image
    height, width = img_ref.shape
    img_reg = cv2.warpPerspective(img_tif, H, (width, height))

    return img_reg, H

# **************************************************************************************************
def homography_rms(H: np.ndarray, m: int, n: int) -> float:
    """
    Find the root mean square displacement of a homagraphy matrix H acting on an image.
    INPUTS:
        H:      a 3x3 homography matrix
        m:      number of rows in the image
        n:      number of colums in the image
    RETURNS
        rms:    The root mean square distance implied by the action of homography matrix H.
    """
    # The shape of the images
    shape_2d = (m, n,)
    # The shape of the coordinate arrays for each image
    shape_3d = (m, n, 3)

    # The X coordinates of the input image
    X1: np.ndarray = np.broadcast_to(np.arange(n).reshape((1, n)), shape_2d)
    # The Y coordinates of the input image
    Y1: np.ndarray = np.broadcast_to(np.arange(m).reshape((m, 1)), shape_2d)
    # The constant entry for the input image
    C1: np.ndarray = np.ones(shape_2d)
    # Build array of (X, Y, 1) for the input image
    P1: np.ndarray = np.zeros(shape_3d)
    P1[:,:,0] = X1
    P1[:,:,1] = Y1
    P1[:,:,2] = C1

    # Apply the homography matrix
    P2: np.ndarray = np.dot(P1, H)

    # Calculate the matrix of pixel distances
    dist_mat: np.ndarray = P2[:,:,0:2] - P1[:,:,0:2]
    # Calculate the root mean square distance
    dist: float = np.sqrt(np.mean(dist_mat * dist_mat))
    return dist

# **************************************************************************************************
def align_one_tif(ic: int, iv: int, img_ref: np.ndarray, verbose: bool) -> float:
    """
    Align one TIF image to the bright field
    INPUTS:
        ic:         Index of the concentration
        iv:         Index of the voltage
        img_ref:    Reference image as a numpy array
        verbose:    Verbosity of output
    RETURNS:
        rms:        Root mean square displacement of the homography
    """

    # Stem of the file names (shared by PNG and TIF)
    stem: str = f'conc{ic:d}_pot{iv:d}'

    # Read image to be aligned
    fname_aln = f'calcs/png_clipped/{stem:s}.png'
    img_aln = cv2.imread(fname_aln, cv2.IMREAD_UNCHANGED)

    # Read TIF image that will be transformed
    fname_tif = f'data/tif/{stem:s}.tif'
    img_tif = cv2.imread(fname_tif, cv2.IMREAD_UNCHANGED)

    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in H.
    img_reg, H = align_image(img_aln=img_aln, img_ref=img_ref, img_tif=img_tif)

    # Write aligned image to disk.
    fname_out = f'calcs/png_aligned/{stem:s}.png'
    if verbose:
        print(f'Saving aligned image : {fname_out:s}')
        print(f'Data type = {img_reg.dtype}')
    cv2.imwrite(fname_out, img_reg)

    # Print estimated homography
    if verbose:
        print("Estimated homography : \n",  H)

    # Return the root mean square distance implied by the homography
    m: int = img_aln.shape[0]
    n: int = img_aln.shape[1]
    rms: float = homography_rms(H=H, m=m, n=n)
    return rms
