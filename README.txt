ID: 207950577


python version: 3.8


platform: pycharm


files to submit:

* ex1_main.py-
	def histEqDemo(img_path: str, rep: int):
		Image histEq display

	def quantDemo(img_path: str, rep: int):
		Image Quantization display

	def main():
		Main test for the functions in ex1_utils and for gamma.

* ex1_utils.py-
	def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
		Reads an image, and returns in converted as requested

	def imDisplay(filename: str, representation: int):
		Reads an image as RGB or GRAY_SCALE and displays it

	def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
		Converts an RGB image to YIQ color space
	def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    		Converts an YIQ image to RGB color space

	def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
   		Equalizes the histogram of an image

	def fix_q(z: np.array, image_hist: np.ndarray) -> np.ndarray:
        	Calculate the new q using wighted average on the histogram
    def fix_z(q: np.array, z: np.array) -> np.array:
            Calculate the new z using the formula from the lecture.
	def findBestCenters(histOrig: np.ndarray, nQuant: int, nIter: int) -> (np.ndarray, np.ndarray):
        	Finding the best nQuant centers for quantize the image in nIter steps or when the error is minimum
	def convertToImg(imOrig: np.ndarray, histOrig: np.ndarray, yiqIm: np.ndarray, arrayQuantize: np.ndarray) ->(np.ndarray, float):
        	Executing the quantization to the original image
	def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
        	Quantized an image in to nQuant colors

* gamma.py-
	def gammaDisplay(img_path: str, rep: int):
    		GUI for gamma correction
	def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        	Gamma correction
	def main():
		Main Test for gamma correction

* testImg1.jpg - dark image, the most pixels in the image are black.

* testImg2.jpg - standard image with many colors.

I desided to take those images becasuse they represent a lot of images in the real world, and I want to examine the goodness of the code. I want to check weather the hsitogramEqualize() works on a dark image and weather the quantizeImage() works on a images with many colors.


answer for the question from section 4.5:
	If a division will have a grey level segment with no pixels, procedure will crash because we will not be able to calculate the weighted average for this segment because we need to divide by the number of pixels for this segment, but in this case the number is zero.