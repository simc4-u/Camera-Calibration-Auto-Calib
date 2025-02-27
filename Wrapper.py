import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import glob
import os

def main():
    imset = []
    path = "D:/Computer vision/Homeworks/4. Auto Calib HW1/Calibration_Imgs/Calibration_Imgs"
    print("os.listdir", os.listdir(path))
    for im in os.listdir(path):
        img = cv2.imread(os.path.join(path, im))
        #img = cv2.resize(img, (640, 480))
        im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Original", im)
        cv2.waitKey(1)
        imset.append(im)
    #Camera intrinsic matrix
    def intrinsic_matrix(homography):
        V = []
        for H in homography:
            h1, h2 = H[:,0], H[:,1]
            v12 = np.array([
                h1[0] * h2[0],
                h1[0] * h2[1] + h1[1] * h2[0],
                h1[1] * h2[1],
                h1[2] * h2[0] + h1[0] * h2[2],
                h1[2] * h2[1] + h1[1] * h2[2],
                h1[2] * h2[2]
            ])

            v11 = np.array([
                h1[0] ** 2, 2 * h1[0] * h1[1], h1[1] ** 2,
                2 * h1[0] * h1[2], 2 * h1[1] * h1[2], h1[2] ** 2
            ])

            v22 = np.array([
                h2[0] ** 2, 2 * h2[0] * h2[1], h2[1] ** 2,
                2 * h2[0] * h2[2], 2 * h2[1] * h2[2], h2[2] ** 2
            ])

            V.append(v12.T)
            V.append((v11 - v22).T)
            print("Vlist",V)
        V = np.array(V)
        print("Varray",V)
        # Solve Vb = 0 using SVD
        _, _, Vt = np.linalg.svd(V)
        b = Vt[-1]  # Eigenvector with smallest eigenvalue
        print("b",b)

        # Reconstruct B matrix
        B = np.array([
            [b[0], b[1], b[3]],
            [b[1], b[2], b[4]],
            [b[3], b[4], b[5]]
        ])

        # Extract intrinsic parameters
        # gamma = 0
        #B[0,1] = 0
        v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
        lambda_ = B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
        alpha = np.sqrt(lambda_ / B[0, 0])
        beta = np.sqrt(lambda_ * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
        gamma = -B[0, 1] * alpha ** 2 * beta / lambda_
        u0 = gamma * v0 / beta - B[0, 2] * alpha ** 2 / lambda_

        # Return camera matrix
        return np.array([
            [alpha, gamma, u0],
            [0, beta, v0],
            [0, 0, 1]
        ])

    #extrinsic parameters
    def extrinsic_matrix(A, H):
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        A_inv = np.linalg.inv(A)
        # Calculate scaling factor lambda
        lambda_ = 1 / np.linalg.norm(np.dot(A_inv, h1))
        r1 = lambda_ * np.dot(A_inv, h1)
        r2 = lambda_ * np.dot(A_inv, h2)
        r3 = np.cross(r1, r2)
        t = lambda_ * np.dot(A_inv, h3)
        # Form rotation matrix
        R = np.column_stack((r1, r2, r3))

        return R , t

    #Corners detection using Chessboard corners
    n_rows = 9
    n_cols = 6
    square_size = 21.7
    wp = np.zeros((n_rows*n_cols,3), np.float32)
    print("wp", wp)
    wp[:,:2] = np.mgrid[1:10,1:7].T.reshape(-1, 2)* square_size# 8 * 5 grid
    print("wp", wp)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    wpoints = []
    opoints = []
    homography = []
    i = 0
    for im in imset:
        ret, corners = cv2.findChessboardCorners(im, (9, 6), None)
        print(ret)
        if ret == True:
            print("corners found")
            wpoints.append(wp)
            corners2 = cv2.cornerSubPix(im, corners,(11,11),(-1,-1),criteria)
            opoints.append(corners2)
            print("corners2", corners2)

            # gethomography
            H, _ = cv2.findHomography(wp[:, :2], corners2.reshape(-1, 2))
            homography.append(H)
            print(f"homography for {i} :", H)

            # Draw and display the corners
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(im, (9, 6), corners2, ret)
            cv2.imshow('img', im)
            output_path = f"D:/Computer vision/Homeworks/4. Auto Calib HW1/corners/corners_{i}.jpg"
            cv2.imwrite(output_path, im)
            print(f"Saved image with corners to: corners_{i}.jpg")
            #cv2.imwrite("corners{i}", im)
            i = i + 1
            print("i", i)
            cv2.waitKey(500)
    print("wpoints", wpoints)

    if len(homography) >= 3:
        camera_matrix = intrinsic_matrix(homography)
        print("\nCamera Intrinsic Matrix:")
        print(camera_matrix)
        print("\nFocal lengths:")
        print(f"fx = {camera_matrix[0, 0]:.2f}")
        print(f"fy = {camera_matrix[1, 1]:.2f}")
        print("\nPrincipal point:")
        print(f"cx = {camera_matrix[0, 2]:.2f}")
        print(f"cy = {camera_matrix[1, 2]:.2f}")
        print("\nSkew:")
        print(f"gamma = {camera_matrix[0, 1]:.2f}")
    else:
        print("Need at least 3 images for calibration!")

    R = []
    best_R = []
    t = []
    #extrinsic parameters
    for i, H in enumerate(homography):
        Rotation, translation = extrinsic_matrix(camera_matrix, H)
        #R.append(Rotation)
        t.append(translation)
        U, _, Vt = np.linalg.svd(Rotation)
        best_rotation = np.dot(U, Vt)
        R.append(best_rotation) #storing best rotation matrix
        if np.linalg.det(best_rotation) < 0:
            Vt[2, :] *= -1
            best_rotation = np.dot(U, Vt)
        best_R.append(best_rotation)
        print(f'Rotation and translation for image{i+1}: {Rotation}, {translation}')
    print("Rotation", R)
    print("translation", t)
    "Intialisation Distortion cofficient"
    k1 = 0
    k2 = 0

    def reprojection_error(wpoints, opoints, camera_matrix, R, t, k1, k2):
        per_image_errors = []
        all_errors =[]
        projected_points = []
        for i in range(len(wpoints)):
            projected = project_pointss(wpoints[i], camera_matrix, R[i], t[i], k1, k2)
            projected_points.append(projected)
            obs_points = opoints[i].reshape(-1, 2)
            # Calculate error for each point
            error = (projected - obs_points).ravel()  # Flatten x,y differences
            image_rms = np.sqrt(np.mean(error ** 2))
            per_image_errors.append(image_rms)
            all_errors.append(error)
        all_errors = np.concatenate(all_errors)
        rms_error = np.sqrt(np.mean(all_errors ** 2))
        return rms_error, per_image_errors


    def project_pointss(points_3d, K, R, t, k1, k2):
        """Project 3D points to 2D with distortion"""
        # Apply extrinsic parameters
        if points_3d.shape[1] == 2:
            points_3d = np.hstack((points_3d, np.zeros((points_3d.shape[0], 1))))
        points_cam = R @ points_3d.T + t.reshape(3, 1)

        # Normalize coordinates
        x = points_cam[0, :] / points_cam[2, :]
        y = points_cam[1, :] / points_cam[2, :]

        # Apply radial distortion
        r2 = x * x + y * y
        distortion = 1 + k1 * r2 + k2 * r2 * r2
        x_distorted = x * distortion
        y_distorted = y * distortion

        # Apply camera matrix
        points_2d = np.vstack([
            K[0, 0] * x_distorted + K[0, 1] * y_distorted + K[0, 2],
            K[1, 1] * y_distorted + K[1, 2]
        ]).T

        return points_2d
    #projection error before optimisation
    rms_error, per_image_errors= reprojection_error(wpoints, opoints, camera_matrix, R, t, k1, k2)
    print(f"Root Mean square error before optimization: {rms_error}")
    print(f"Per-image errors before optimization: {per_image_errors}")


    def params_conversion(params, len):
        fx, fy, cx, cy, k1, k2 = params[:6]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        Rs = []
        ts = []
        for i in range(len):
            idx = 6 + i * 6
            rvec = params[idx:idx + 3]
            R, _ = cv2.Rodrigues(rvec)
            t = params[idx + 3:idx + 6]
            Rs.append(R)
            ts.append(t)

        return K, k1, k2, Rs, ts

    def reprojection_errors(params,  wpoints, opoints ):
        n_img = len(wpoints)
        #print("wplen", n_img)
        K, k1, k2, Rs, ts = params_conversion(params, n_img)
        all_error = []
        for i in range(n_img):
            projected_points = project_pointss(
                wpoints[i],
                K,
                Rs[i],
                ts[i],
                k1,
                k2
            )
            img_points = opoints[i].reshape(-1, 2)
            error = (projected_points - img_points).ravel()
            #print(f"error of image{i+1}: {np.mean(error)}")
            all_error.append(error)
        #print(f"all_error_shape:{np.concatenate(all_error).shape}")
        #print(f"total error {np.mean(np.concatenate(all_error))}")
        return np.concatenate(all_error)

    def optimization (wpoints, opoints, camera_matrix, R, t, k1=0, k2=0):
        n_img = len(wpoints)
        params_init = np.zeros(6 + 6*n_img)
        #set intrinsic parameters
        params_init[0] = camera_matrix[0, 0]  # fx
        params_init[1] = camera_matrix[1, 1]  # fy
        params_init[2] = camera_matrix[0, 2]  # cx
        params_init[3] = camera_matrix[1, 2]  # cy
        params_init[4] = k1  # k1
        params_init[5] = k2  # k2

        #Set intrinsic parameters
        for i in range(n_img):
            idx = 6 + i * 6
            rvec, _ = cv2.Rodrigues(R[i])
            params_init[idx:idx + 3] = rvec.ravel()
            params_init[idx + 3:idx + 6] = t[i].ravel()

        result = least_squares(
            reprojection_errors,
            params_init,
            args=(wpoints, opoints),
            method='lm',
            max_nfev=100,  # maximum number of function evaluations
            ftol=1e-8,  # function tolerance
            xtol=1e-8  # parameter tolerance
        )

        # Extract optimized parameters
        K_opt, k1_opt, k2_opt, Rs_opt, ts_opt = params_conversion(result.x, n_img)
        print(f"error: {result.fun}")
        print(f"shape: {result.fun.shape}")
        # Compute final error (RMS)
        final_error = np.sqrt(np.mean(result.fun ** 2))

        return K_opt, [k1_opt, k2_opt], Rs_opt, ts_opt, final_error

    K_opt, dist_opt, R_opt, t_opt, final_error = optimization(
        wpoints,
        opoints,
        camera_matrix,
        R,
        t,
        k1=0,
        k2=0
    )
    """reprojection eror after optimization"""
    rms_error_opt, per_image_errors_opt = reprojection_error(
        wpoints,
        opoints,
        K_opt,  # Optimized camera matrix
        R_opt,  # Optimized rotation matrices
        t_opt,  # Optimized translation vectors
        dist_opt[0],  # Optimized k1
        dist_opt[1]  # Optimized k2
    )

    print("\nOptimized Camera Matrix (Levenberg-Marquardt):")
    print(K_opt)
    print("\nOptimized Distortion Coefficients:")
    print(f"k1 = {dist_opt[0]:.6f}")
    print(f"k2 = {dist_opt[1]:.6f}")
    print(f"R ", R_opt)
    print(f"t ", t_opt)
    print("\nReprojection errors after optimization:")
    print(f"mean_error_after_optimization: {rms_error_opt}")
    print(f"per_image_errors_after_optimization: {per_image_errors_opt}")

    "undistorted images"
    for i, (img, wp, imgpoint) in enumerate(zip(imset, wpoints, opoints)):
        h, w = img.shape[:2]
        #print("\nDistortion Coefficients:", dist_array)
        mtx, roi = cv2.getOptimalNewCameraMatrix(K_opt,   np.array([dist_opt[0], dist_opt[1], 0, 0, 0]), (w, h), 1, (w, h))
        undist_img = cv2.undistort(img, K_opt,   np.array([dist_opt[0], dist_opt[1], 0, 0, 0]), None, mtx )
        #Crop the image
        x, y, w, h = roi
        undist_img = undist_img[y:y + h, x:x + w]
        cv2.imshow("undistorted image", undist_img)
        output_pat = f"D:/Computer vision/Homeworks/4. Auto Calib HW1/undistortedimages/undistortedimg_{i}.jpg"
        cv2.imwrite(output_pat, undist_img)

        #Convert to color
        undist_color = cv2.cvtColor(undist_img, cv2.COLOR_GRAY2BGR)


        #opoints in corners
        corners = imgpoint.reshape(-1, 2)
        projected = project_pointss(
            wp,
            mtx,
            R_opt[i],
            t_opt[i],
            dist_opt[0],
            dist_opt[1]
        )

        # Adjustment for cropping
        corners_adjusted = corners.copy()
        corners_adjusted[:, 0] -= x
        corners_adjusted[:, 1] -= y

        projected_adjusted = projected.copy()
        projected_adjusted[:, 0] -= x
        projected_adjusted[:, 1] -= y

        # Draw reprojected corners red
        for pt in projected_adjusted:
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(undist_color, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), 2)

        output_pa = f"D:/Computer vision/Homeworks/4. Auto Calib HW1/reprojected/reprojected_{i}.jpg"
        cv2.imwrite(output_pa, undist_color)
        print(f"saving image undist_color{i+1}")

        #cv2.imwrite('calibresult.png', undist_img)



    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


