import os
import cv2
import joblib
import numpy as np
import pandas as pd
from get_img import get_image

clf = joblib.load('iris_stress_model.pkl')

folder_path = "DATA_IMAGES"

image_path = os.path.join(folder_path, get_image(folder_path))

try:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (280, 320))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    blurred = cv2.GaussianBlur(edges, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[:2]

        pupil_circle = circles[0]
        iris_circle = circles[1]

        iris_x, iris_y = iris_circle[0], iris_circle[1]

        iris_radius = int(iris_circle[2] * 2.2)

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

            input_data = pd.DataFrame({'iris_diameter': [iris_diameter],
                                        'pupil_diameter': [pupil_diameter],
                                        'iris_to_pupil_ratio': [iris_to_pupil_ratio]})

            predicted_stress_level = clf.predict(input_data)[0]

            stress_levels = {
                1: 'low stress',
                2: 'moderate stress',
                3: 'high stress',
                4: 'anxious',
                5: ' very anxious'
            }

            stress_level_text = stress_levels.get(predicted_stress_level, 'unknown')

            print("Iris Diameter:", iris_diameter)
            print("Pupil Diameter:", pupil_diameter)
            print("Iris to Pupil Ratio:", iris_to_pupil_ratio)
            print("Predicted Stress Level:", stress_level_text)
        else:
            print("Iris not properly detected.")
    else:
        print("Pupil not detected in the image.")
except Exception as e:
    print("Error:", e)
    print("Invalid image path. Please try again.")