import numpy as np
import cv2

def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    return edges

def find_the_min_rect(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    return rect, box

if __name__ == "__main__":
    image = cv2.imread("/home/hcis/YongZhe/CropRGB.png")
    
    preprocessing_img = preprocessing(image)
    # cv2.imshow("preprocessing image", preprocessing_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    rect, box = find_the_min_rect(preprocessing_img)
    (center_x, center_y), (width, height), angle = rect

    if width < height:
        yaw_angle = angle + 90  # turn into [0, 180)
    else:
        yaw_angle = angle

    print(f"Yaw angle of socket: {yaw_angle:.2f} degrees")

    drawn_image = image.copy()
    cv2.drawContours(drawn_image, [box], 0, (0, 0, 255), 2)

    cv2.imshow("Detected Socket and Yaw", drawn_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
