import cv2
import numpy as np

def get_dark_channel(image, window_size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmosphere(image, dark_channel):
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    num_pixels = flat_dark.size
    num_brightest = int(max(num_pixels * 0.001, 1))
    indices = np.argpartition(-flat_dark, num_brightest)[:num_brightest]
    atmosphere = np.max(flat_image[indices], axis=0)
    return atmosphere

def get_transmission(image, atmosphere, omega=0.95, window_size=15):
    norm_image = image / atmosphere
    transmission = 1 - omega * get_dark_channel(norm_image, window_size)
    return transmission

def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q

def recover_image(image, transmission, atmosphere, t0=0.1):
    transmission = np.clip(transmission, t0, 1)
    J = (image - atmosphere) / transmission[:, :, np.newaxis] + atmosphere
    J = np.clip(J, 0, 255)
    return J.astype(np.uint8)

image = cv2.imread("foggy_image.jpg").astype(np.float64)

dark_channel = get_dark_channel(image / 255.0)
atmosphere = get_atmosphere(image, dark_channel)
transmission = get_transmission(image, atmosphere)

gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
refined_transmission = guided_filter(gray, transmission, r=60, eps=1e-3)

dehazed_image = recover_image(image, refined_transmission, atmosphere)

cv2.imwrite("fog_removed.jpg", dehazed_image)
cv2.imshow("Original Image", image.astype(np.uint8))
cv2.imshow("Dehazed Image", dehazed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
