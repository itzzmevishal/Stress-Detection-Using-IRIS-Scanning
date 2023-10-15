import os
import cv2
import numpy as np
import pandas as pd

data_list = []

image_folder = 'DATA_IMAGES/'

allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

for filename in os.listdir(image_folder):
    if any(filename.lower().endswith(ext) for ext in allowed_extensions):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, threshold1=30, threshold2=100)

        blurred = cv2.GaussianBlur(edges, (5, 5), 0)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)

        if circles is not None:
            circles = np.uint16(np.around(circles))

            circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[:2]

            pupil_circle = circles[0]
            iris_circle = circles[1]

            iris_x, iris_y = pupil_circle[0], pupil_circle[1]

            iris_radius = int(pupil_circle[2] * 2.2)

            mask = np.zeros_like(gray)
            cv2.circle(mask, (iris_x, iris_y), iris_radius, 255, -1)

            iris_roi = cv2.bitwise_and(blurred, blurred, mask=mask)

            _, iris_thresholded = cv2.threshold(iris_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            iris_contours, _ = cv2.findContours(iris_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if iris_contours:
                iris_contour = max(iris_contours, key=cv2.contourArea)

                iris_diameter = round(cv2.minEnclosingCircle(iris_contour)[1] * 2, 2)
                pupil_diameter = round(pupil_circle[2] * 2, 2)

                iris_to_pupil_ratio = round(iris_diameter / pupil_diameter, 2)

                if iris_to_pupil_ratio < 1.9:
                    stress_level = 1
                elif 1.91 <= iris_to_pupil_ratio <= 2.0:
                    stress_level = 2
                elif 2.01 <= iris_to_pupil_ratio <= 2.1:
                    stress_level = 3
                elif 2.11 <= iris_to_pupil_ratio <= 2.2:
                    stress_level = 4
                else:
                    stress_level = 5

                data_list.append({"image": filename, "iris_diameter": iris_diameter, "pupil_diameter": pupil_diameter, "iris_to_pupil_ratio": iris_to_pupil_ratio, "stress_level": stress_level})
            else:
                print(f"Iris not properly detected in {filename}.")
        else:
            print(f"Pupil not detected in {filename}.")

data = pd.DataFrame(data_list)

data.to_excel('iris_data.xlsx', index=False)
print("Data saved to iris_stress.xlsx")
