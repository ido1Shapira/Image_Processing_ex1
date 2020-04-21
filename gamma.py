"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import *

def on_trackbar(val):
    pass

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    img = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:  # for gray image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    title_window = 'Gamma Display'
    trackbar_name = 'Gamma'
    cv2.namedWindow(title_window)
    cv2.createTrackbar(trackbar_name, title_window, 0, 100, on_trackbar)
    while True:
        gamma = cv2.getTrackbarPos(trackbar_name, title_window)
        gamma = gamma/100 * (2 - 0.01)
        gamma = 0.01 if gamma == 0 else gamma
        newImg = adjust_gamma(img, gamma)
        cv2.imshow(title_window, newImg)
        k = cv2.waitKey(1000)
        if k == 27:  # esc button
            break
        if cv2.getWindowProperty(title_window, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """
        Gamma correction
        :param image: the original image
        :param gamma: the gamma number
        :return: the new image after the gamma operation
        """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def main():
    gammaDisplay('sample_image.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
