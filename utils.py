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


def run_sift(img1, img2, SHOW_PLOT=False):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    res1 = cv.drawKeypoints(
        img1, kp1, img1.copy(), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    sift = cv.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img2, None)
    res2 = cv.drawKeypoints(
        img2, kp2, img2.copy(), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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
    if SHOW_PLOT:
        plot([img1, img2, out, img3], nrows=2, ncols=2)
    return out


def drawKeyPts(image, keypoints, col, thickness):
    for point in keypoints:
        x = int(point.pt[0])
        y = int(point.pt[1])
        size = int(point.size)
        cv.circle(image, (x, y), size, col,
                  thickness=thickness, lineType=8, shift=0)
    return image


def extract_fast_features(reference, target, reference_card, input_image, USE_CONTOURS=True):
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
        min_area = 5500
        max_area = 7000
        image_number = 0
        for c in reference_contours:
            area = cv2.contourArea(c)
            # if area > 500:
            #     print(area)
            if area > min_area and area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                points.append((x, y))
                points.append((x+w, y))
                points.append((x, y+h))
                points.append((x+w, y+h))

                padding = 6
                ROI = reference_card[y+padding:y +
                                     h-padding, x+padding:x+w-padding]
                cv2.imwrite(f"boxes/Box_{image_number}.png", ROI)
                reference_with_rects = cv2.rectangle(reference_card, (x, y),
                                                     (x + w, y + h), (255, 0, 0), 2)
                image_number += 1
        image_number = 0
        for c in target_contours:
            area = cv2.contourArea(c)
            # if area > 500:
            #     print(area)
            if area > min_area and area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                points.append((x, y))
                points.append((x+w, y))
                points.append((x, y+h))
                points.append((x+w, y+h))

                padding = 6
                ROI = input_image[y+padding:y +
                                  h-padding, x+padding:x+w-padding]
                cv2.imwrite(f"target_boxes/Box_{image_number}.png", ROI)
                target_with_rects = cv2.rectangle(input_image, (x, y),
                                                  (x + w, y + h), (255, 0, 0), 2)
                image_number += 1

        plot([reference_with_rects, target_with_rects])
    else:
        fast = cv.FastFeatureDetector_create()
        fast.setNonmaxSuppression(False)
        fast.setThreshold(32)

        kp1 = fast.detect(reference, None)
        referenceWithCircles = drawKeyPts(
            reference_card.copy(), kp1, (0, 0, 255), 2)

        kp2 = fast.detect(target, None)
        targetWithCircles = drawKeyPts(input_image.copy(), kp2, (0, 0, 255), 2)

        plot([referenceWithCircles, targetWithCircles])


def remove_noise_before_keypoint_detecton(image, use_ui=False):
    out = image.copy()
    kernel_size = 20
    iterations = 4
    if use_ui == False:
        return out

    font = cv2.FONT_HERSHEY_SIMPLEX

    def update_kernel_size(x):
        kernel_size = x

    def update_iterations(x):
        kernel_size = x

    cv2.namedWindow('controls')
    cv2.createTrackbar('kernel_size', 'controls', 3, 100, update_kernel_size)
    cv2.createTrackbar('iterations', 'controls', 1, 10, update_iterations)

    while(True):
        k_size = int(cv2.getTrackbarPos('kernel_size', 'controls'))
        it = int(cv2.getTrackbarPos('iterations', 'controls'))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

        out = cv2.morphologyEx(
            image, cv2.MORPH_CLOSE, kernel, iterations=it)
        # out = cv2.morphologyEx(
        #     image, cv2.MORPH_OPEN, kernel, iterations=it)

        out = cv2.putText(
            out, f"Kernel Size: {k_size} -- Iterations: {it}", (30, 50), font, 0.9, (255, 255, 255), 1)

        cv2.imshow('controls', out)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    return out


def angle_cos(p0, p1, p2):
    # https://github.com/opencv/opencv/blob/master/modules/python/test/test_squares.py
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def extract_all_points(original, after_threshold):

    image = original.copy()
    contours = cv2.findContours(
        after_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    squares = []
    # https://coderedirect.com/questions/493257/advanced-square-detection-with-connected-region
    for cnt in contours:
        print("(=====)")
        print(cnt)
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        print(cnt_len, cnt)
        print("(=====)")

        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max(
                [angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in range(4)])
            if max_cos < 0.1:
                squares.append(cnt)
                cv2.circle(image, cnt[0], 8, (0, 0, 255), -1)
                cv2.circle(image, cnt[1], 8, (0, 255, 0), -1)
                cv2.circle(image, cnt[2], 8, (255, 0, 0), -1)
                cv2.circle(image, cnt[3], 8, (255, 255, 0), -1)
        continue


    # for c in contours:
    #     no_of_points = len(c)
    #     if(no_of_points < 4):
    #         continue
    #     elif(no_of_points == 4):
    #         extLeft, extRight = c.reshape(-1, 2)[:2]
    #         extTop, extBot = c.reshape(-1, 2)[-2:]
    #     else:
    #         extLeft, extRight = c[c.sum(
    #             axis=2).argmin()][0], c[c.sum(axis=2).argmax()][0]
    #         extTop, extBot = [0, 0], [0, 0]
    #         continue

    #         increasing_noise = np.linspace(
    #             0., .9, num=len(c[:, :, 0]), dtype='float32')
    #         decreasing_noise = np.linspace(.9, 0., num=len(
    #             c[:, 0]), dtype='float32')

    #         extLeft = tuple(c[c[:, :, 0].argmin()][0])
    #         extRight = tuple(c[c[:, :, 0].argmax()][0])
    #         extTop = tuple(c[c[:, :, 1].argmin()][0])
    #         extBot = tuple(c[c[:, :, 1].argmax()][0])

            # if i < 1:
            #     i += 1
            #     idx = c[:, 1].argmax()
            #     print(c, idx, c[idx])
            # print(c, c[:, :, 0], np.argmin(c[:, :, 0] + increasing_noise))
            # print(len(c), i)
            # print(extLeft, extRight, extTop, extBot)

            # topLeft = tuple(c[np.argmin(c[:, :, 0] + increasing_noise)][0])
            # topRight = tuple(c[np.argmin(c[:, :, 0] + decreasing_noise)][0])
            # bottomLeft = tuple(c[np.argmin(c[:, :, 1] + increasing_noise)][0])
            # bottomRight = tuple(c[np.argmax(c[:, :, 1] + increasing_noise)][0])

            # if (cv2.norm(extLeft - extRight) > 300):
            #     continue
            # cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
            # cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
            # cv2.circle(image, extRight, 8, (0, 255, 0), -1)
            # cv2.circle(image, extTop, 8, (255, 0, 0), -1)
            # cv2.circle(image, extBot, 8, (255, 255, 0), -1)

    cv2.imshow("Contours", image)
    cv2.waitKey(0)
    # print(extTop, extRight, extBot, extLeft)
    # new_c = [(extLeft, extTop), (extLeft, extBot),
    #          (extRight, extBot), (extRight, extTop)]
    # print(new_c)
    # i += 1
    # if i > 5:
    #     break
