import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt

from config import WIDTH, HEIGHT, IMAGE_DIMS


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_threshold(image, low=127, high=255, mode=cv2.THRESH_BINARY_INV):
    return cv2.threshold(image, low, high, mode)


def resize(image):
    '''
        Reshapes and rotates to fit dims of reference
        If uses different aspect ratio - pad with white pixels
    '''
    h, w, c = image.shape
    if(w < h):
        image = cv2.transpose(image)
        image = cv2.flip(image, flipCode=0)

    return cv2.resize(image, IMAGE_DIMS, cv2.INTER_AREA)


def pad(image, size=32, background=[255, 255, 255]):
    return cv2.copyMakeBorder(
        image.copy(), size, size, size, size, cv2.BORDER_CONSTANT, value=WHITE)


def median_blur(image, kernel_size=7):
    ''' 
        Too much noise - apply median blur
        Median blur helps eliminate salt and pepper noise - and also preserves edges
    '''
    return cv.medianBlur(image, kernel_size)


def apply_opening(image, kernel_size=40, iterations=1):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations)


def apply_closing(image, kernel_size=40, iterations=1):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations)


def run_split_visualizer(img1, img2, name):
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    cv2.imshow('Original', img1)
    cv2.imshow(name, img2)

    cv2.resizeWindow('Original', WIDTH, HEIGHT)
    cv2.resizeWindow(name, WIDTH, HEIGHT)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_visualizer(img1, img2, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # x, y = img2.shape
    x, y, z = img2.shape
    img1 = cv2.resize(img1, (y, x))
    print(img1.shape, img2.shape)
    stacked = np.hstack((img1, img2))
    cv2.imshow(name, stacked)
    cv2.resizeWindow(name, WIDTH, HEIGHT)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calibrate_threshold(image, use_ui=False):
    low = 127
    high = 255

    if use_ui == False:
        return low, high

    font = cv2.FONT_HERSHEY_SIMPLEX

    def update_low_threshold(x):
        low = x

    def update_high_threshold(x):
        high = x

    cv2.namedWindow('controls')
    cv2.createTrackbar('low', 'controls', 127, 255, update_low_threshold)
    cv2.createTrackbar('high', 'controls', 255, 255, update_high_threshold)

    while(True):
        low = int(cv2.getTrackbarPos('low', 'controls'))
        high = int(cv2.getTrackbarPos('high', 'controls'))
        (thresh, image_with_threshold) = cv2.threshold(
            image, low, high, cv2.THRESH_BINARY_INV)
        image_with_threshold = cv2.putText(
            image_with_threshold, f"Low: {low} -- High: {high}", (30, 330), font, 0.7, (255, 255, 255), 1)

        cv2.imshow('controls', image_with_threshold)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    return low, high


def plot(images, nrows=1, ncols=2, figsize=(16, 6)):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = [ax] if nrows * ncols == 1 else ax.ravel()
    fig.patch.set_visible(False)
    for i in range(nrows * ncols):
        axes[i].axis('off')
    for i in range(len(images)):
        axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.show()


def run_orb(img1, img2):

    MIN_MATCH_COUNT = 10
    orb = cv.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20,
                        scaleFactor=1.2, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,     # 20
                        multi_probe_level=1)  # 2

    # Then set number of searches. Higher is better, but takes longer
    search_params = dict(checks=100)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for match in matches:
        if(len(match) != 2):
            continue
        m, n = match
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        matchesMask = mask.ravel().tolist()

        try:
            h, w, d = img1.shape
        except:
            h, w = img1.shape

        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                         ).reshape(-1, 1, 2)

        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plot([img3], nrows=1, ncols=1)


def run_sift(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    res1 = cv.drawKeypoints(
        img1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    sift = cv.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img2, None)
    res2 = cv.drawKeypoints(
        img2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # run_visualizer(res1, res2, 'SIFT')

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))

    out = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
    plot([img1, img2, out, img3], nrows=2, ncols=2)


def drawKeyPts(im, keyp, col, th):
    for curKey in keyp:
        x = np.int(curKey.pt[0])
        y = np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv.circle(im, (x, y), size, col, thickness=th, lineType=8, shift=0)
    return im


def extract_fast_features(reference, target, reference_card, USE_CONTOURS=True):
    if USE_CONTOURS:
        reference_contours = cv2.findContours(
            reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reference_contours = reference_contours[0] if len(
            reference_contours) == 2 else reference_contours[1]

        target_contours = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_contours = target_contours[0] if len(
            target_contours) == 2 else target_contours[1]

        # https://stackoverflow.com/a/57193144/5837671
        points = []
        min_area = 6000
        max_area = 7000
        image_number = 0
        for c in reference_contours:
            area = cv2.contourArea(c)
            if area > 500:
                print(area)
            if area > min_area and area < max_area:
                print(area)
                x, y, w, h = cv2.boundingRect(c)
                points.append((x, y))
                points.append((x+w, y))
                points.append((x, y+h))
                points.append((x+w, y+h))

                padding = 6
                ROI = reference_card[y+padding:y +
                                     h-padding, x+padding:x+w-padding]
                cv2.imwrite(f"boxes/Box_{image_number}.png", ROI)
                reference_with_rects = cv2.rectangle(reference, (x, y),
                                                     (x + w, y + h), (255, 0, 0), 2)
                image_number += 1

        plot([reference_with_rects, target])
    else:
        fast = cv.FastFeatureDetector_create()
        fast.setNonmaxSuppression(False)
        fast.setThreshold(32)

        kp1 = fast.detect(reference, None)
        referenceWithCircles = drawKeyPts(
            reference.copy(), kp1, (0, 0, 255), 2)

        kp2 = fast.detect(target, None)
        targetWithCircles = drawKeyPts(target.copy(), kp2, (0, 0, 255), 2)

        plot([referenceWithCircles, targetWithCircles])
