#David Poole CS410p Computer Vision winter 2020 homework 2 
import cv2 as cv
import sys
import numpy as np
import random

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3,
                              max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''

    best_H = None
    most_inliers = []
    # Loop large number of times
    for i in range(0, max_num_trial):
        # Randomly select 4 corresponding points (total 8)
        size = len(list_pairs_matched_keypoints)
        lst = []
        while len(lst) < 4:
            r = random.randrange(size)
            if list_pairs_matched_keypoints[r] not in lst:
                lst.append(list_pairs_matched_keypoints[r])


        A =  []
        for j in range(4):
           x_1 = lst[j][0][0]
           y_1 = lst[j][0][1]
           x_2 = lst[j][1][0]
           y_2 = lst[j][1][1]

           r_a = np.asarray([x_1, y_1, 1., 0, 0, 0, -x_1 * x_2, -y_1 * x_2, -x_2]) ##
           r_b = np.asarray([0, 0, 0, x_1, y_1, 1., -x_1 * y_2, -y_1 * y_2, -y_2])
           A.append(r_a)
           A.append(r_b)

        A = np.asarray(A)

        U, S, V = np.linalg.svd(np.array(A, np.float32))
        curr_H = V[-1, :].reshape(3, 3)
        curr_inliers = []
        for k in range(len(list_pairs_matched_keypoints)):
            distance = calculateEuclidianD(list_pairs_matched_keypoints[k],curr_H)
            if distance < threshold_reprojtion_error:
                curr_inliers.append(list_pairs_matched_keypoints[k])

        if len(curr_inliers) > len(list_pairs_matched_keypoints) * threshold_ratio_inliers:
            best_H = curr_H
            return best_H
        else:
            if(len(curr_inliers)>len(most_inliers)):
                best_H = curr_H
                most_inliers = curr_inliers

    return best_H
def calculateEuclidianD(keypoint_mathes,H):

    x_1 = keypoint_mathes[0][0]
    y_1 = keypoint_mathes[0][1]
    x_2 = keypoint_mathes[1][0]
    y_2 = keypoint_mathes[1][1]

    p1 = np.transpose(np.matrix([x_1,y_1,1]))
    estimate1 = np.dot(H,p1)
    estimate1 = (1/estimate1.item(2))*estimate1


    p2 = np.transpose(np.matrix([x_2,y_2,1]))

    error = np.linalg.norm(p2-estimate1)

    return error
def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================
    # extract image 1
    gray_img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    gray_img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp1,des1 = sift.detectAndCompute(gray_img_1, None)
    kp2,des2= sift.detectAndCompute(gray_img_2, None)

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    list_pairs_matched_keypoints = []
    for x in range(des1.shape[0]):
        closest1 = 10000
        closest2 = 100000
        match_index = -1
        for y in range(des2.shape[0]):
            distance =  np.linalg.norm(des1[x] - des2[y]) # will be faster
            if (distance < closest2):
                if distance < closest1:
                    closest2 = closest1
                    closest1 = distance
                    match_index = y
                else:
                    closest2 = distance

        if closest1 / closest2 < ratio_robustness:
            list_pairs_matched_keypoints.append([[kp1[x].pt[0],kp1[x].pt[1]],
                                                 [kp2[match_index].pt[0],kp2[match_index].pt[1]]])

    return list_pairs_matched_keypoints
def ex_warp_blend_crop_image(img_1, H_1, img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''

    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    # to be completed ...
    inv_h = np.linalg.inv(H_1)
    h = img_2.shape[0]*3
    w = img_2.shape[1]*3
    img_panorama = np.zeros((h,w,3))

    left_bottom_corner = []
    left_top_corner = []
    mask = np.zeros((h,w,1))
    crop_mask = np.zeros((h,w,1))
    for x in range (0,w):
        for y in range(0,h):
            destination_cor_h = np.dot(inv_h,np.asarray([x-img_2.shape[1],y-img_2.shape[0],1]))
            source_cor_h = (1/destination_cor_h[2])* destination_cor_h
            source_cor_u, source_cor_v = source_cor_h[0], source_cor_h[1]
            if img_1.shape[1]-1.0> source_cor_u >= 0 and img_1.shape[0]-1.0 > source_cor_v >= 0:
                cor_u_1 = int((source_cor_u//1)+1)
                cor_v_1 = int((source_cor_v//1)+1)
                cor_u = int(source_cor_u//1)
                cor_v = int(source_cor_v//1)
                cor_dif_u = source_cor_u - (source_cor_u//1)
                cor_dif_v = source_cor_v - (source_cor_v//1)
                resamp_val = (1-cor_dif_u)*(1-cor_dif_v)*img_1[cor_v][cor_u]+cor_dif_u*(1-cor_dif_v)*img_1[cor_v][cor_u]+cor_dif_u*cor_dif_v*img_1[cor_v_1][cor_u_1]+(1-cor_dif_u)*cor_dif_v* img_1[cor_v_1][cor_u]
                img_panorama[y][x] = resamp_val
                crop_mask[y][x] +=1
                mask[y][x]+=1
                left_bottom_corner.append(x)
                left_bottom_corner.append(y)


    # ===== blend images: average blending
    for x in range(img_2.shape[1]):
        for y in range(img_2.shape[0]):
            img_panorama[y+img_2.shape[0]][x+img_2.shape[1]]+= img_2[y][x]
            mask[y+img_2.shape[0]][x+img_2.shape[1]]+=1


    new_height = []
    new_width = []
    for x in range (0,w-1):
        for y in range(0,h-1):
            if (mask[y][x] == 1):
                new_width.append(x)
                new_height.append(y)
            if mask[y][x]==2:
                img_panorama[y][x] = img_panorama[y][x]/2

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    # to be completed ...

    highest_y = new_height[np.argmax(new_height)]
    lowest_y = new_height[np.argmin(new_height)]
    length_of_width = (len(new_width))
    final_width = new_width[length_of_width-1] - new_width[0]
    final_height = highest_y - lowest_y
    final_canvas = np.zeros((final_height,final_width,3))
    for x in range (0,final_width):
        for y in range (0,final_height):
            final_canvas[y][x]= img_panorama[y+lowest_y][x+new_width[0]]

    img_panorama = final_canvas

    return img_panorama


def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1, H_1=H_1, img_2=img_2)

    return img_panorama


if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW2: image stitching')
    print('==================================================')
    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]

    # ===== read 2 input images
    img_1 = cv.imread(path_file_image_1)
    img_2 = cv.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))
