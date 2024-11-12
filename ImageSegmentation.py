import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel
from PyQt5.uic import loadUi

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)

        # Connect the button 'loadimg' with the load_image function
        self.loadimg.clicked.connect(self.load_image)
        self.Image = None
        self.grayscalebtn.clicked.connect(self.Grayscale)
        self.cannybtn.clicked.connect(self.Canny)
        self.watershedbtn.clicked.connect(self.WatershedSegmentation)

    def load_image(self):
        # Open file dialog to select image
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', r"<Default Path>",
                                                   "Image files (*.jpg *.png *.bmp *jpeg)")

        if file_name:
            # Read the image using OpenCV
            self.Image = cv2.imread(file_name)

            # Convert the image to RGB format (PyQt5 expects RGB images, OpenCV uses BGR)
            image_rgb = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)

            # Get image dimensions and create QImage
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

            # Set the QImage in a QLabel (label1) as a QPixmap
            self.label1.setPixmap(QtGui.QPixmap.fromImage(q_img))
            self.label1.setScaledContents(True)

    def Grayscale(self):
        if self.Image is not None:
            gray_image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

            # Tampilkan gambar grayscale
            height, width = gray_image.shape
            bytes_per_line = width
            q_img_gray = QtGui.QImage(gray_image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)

            self.label2.setPixmap(QtGui.QPixmap.fromImage(q_img_gray))
            self.label2.setScaledContents(True)

    def Canny(self):
        if self.Image is not None:
            gray_image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

            # Noise Reduction
            gaussian_kernel = (1.0 / 57) * np.array(
                [[0, 1, 2, 1, 0],
                 [1, 3, 5, 3, 1],
                 [2, 5, 9, 5, 2],
                 [1, 3, 5, 3, 1],
                 [0, 1, 2, 1, 0]])
            blurred_image = cv2.filter2D(gray_image, -1, gaussian_kernel)

            # Gradient calculation
            sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            gradient_direction = np.arctan2(sobel_y, sobel_x) * 180.0 / np.pi

            angle = gradient_direction
            angle[angle < 0] += 180

            H, W = gradient_magnitude.shape
            Z = np.zeros_like(gradient_magnitude)

            # Part 3: Non-Maximum Suppression
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    try:
                        q = 255
                        r = 255

                        # angle 0
                        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                            q = gradient_magnitude[i, j + 1]
                            r = gradient_magnitude[i, j - 1]

                        # angle 45
                        elif (22.5 <= angle[i, j] < 67.5):
                            q = gradient_magnitude[i + 1, j - 1]
                            r = gradient_magnitude[i - 1, j + 1]

                        # angle 90
                        elif (67.5 <= angle[i, j] < 112.5):
                            q = gradient_magnitude[i + 1, j]
                            r = gradient_magnitude[i - 1, j]

                        # angle 135
                        elif (112.5 <= angle[i, j] < 157.5):
                            q = gradient_magnitude[i - 1, j - 1]
                            r = gradient_magnitude[i + 1, j + 1]

                        if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                            Z[i, j] = gradient_magnitude[i, j]
                        else:
                            Z[i, j] = 0

                    except IndexError as e:
                        pass

            img_non_max = Z.astype("uint8")

            # Part 4: Hysteresis Thresholding
            weak_threshold = 100
            strong_threshold = 150

            img_canny1 = np.zeros_like(img_non_max)

            for i in range(H):
                for j in range(W):
                    if img_non_max[i, j] > strong_threshold:
                        img_canny1[i, j] = 255
                    elif weak_threshold <= img_non_max[i, j] <= strong_threshold:
                        img_canny1[i, j] = 128  # Mark potential edges

            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    if img_canny1[i, j] == 128:
                        try:
                            if np.max(img_canny1[i - 1:i + 2, j - 1:j + 2]) == 255:
                                img_canny1[i, j] = 255
                            else:
                                img_canny1[i, j] = 0
                        except IndexError as e:
                            pass

            img_canny2 = img_canny1.astype("uint8")

            plt.figure(figsize=(6, 4))
            plt.imshow(blurred_image, cmap='gray')
            plt.title('Noise Reduction (Blurred)')
            plt.show()

            plt.figure(figsize=(6, 4))
            plt.imshow(gradient_magnitude, cmap='gray')
            plt.title('Gradient Magnitude (Sobel)')
            plt.show()

            plt.figure(figsize=(6, 4))
            plt.imshow(img_non_max, cmap='gray')
            plt.title('Non-Maximum Suppression')
            plt.show()

            plt.figure(figsize=(6, 4))
            plt.imshow(img_canny1, cmap='gray')
            plt.title('Hysteresis Thresholding (Part 1)')
            plt.show()

            plt.figure(figsize=(6, 4))
            plt.imshow(img_canny2, cmap='gray')
            plt.title('Hysteresis Thresholding (Part 2 - Final Edges)')
            plt.show()

    def WatershedSegmentation(self):
        if self.Image is not None:
            # Step 1: Convert the image to grayscale and apply Otsu thresholding
            gray_image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
            ret, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Step 2: Noise removal using morphological opening
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            #closing_img = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Step 3: Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Step 4: Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Step 5: Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Step 6: Marker labeling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Step 7: Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Step 8: Mark the region of unknown with zero
            markers[unknown == 255] = 0

            # Step 9: Apply the watershed algorithm on the original image
            img = self.Image
            markers = cv2.watershed(img, markers)
            img[markers == -1] = [255, 0, 0]  # Mark watershed boundaries with red

            num_segments = len(np.unique(markers))

            if num_segments > 3:
                status_label = "Layak digunakan"
            else:
                status_label = "Wadah Tidak Aman"

            plt.figure(figsize=(6, 4))
            plt.imshow(img, cmap='gray')
            plt.title('Cek Hasil : ')

            # Menambahkan teks status_label ke dalam gambar
            plt.text(0.5, -0.1, status_label, ha='center', va='center', fontsize=12,
                     color='white', bbox=dict(facecolor='black', alpha=0.7),
                     transform=plt.gca().transAxes)

            plt.show()

            plt.figure(figsize=(6, 4))
            plt.imshow(dist_transform, cmap='gray')  # Menggunakan cmap 'gray' untuk grayscale
            plt.title('Distance Transform')
            plt.axis('off')  # Menghilangkan sumbu
            plt.show()

            # Display Otsu thresholding result in label3
            height, width = otsu_thresh.shape
            bytes_per_line = width
            q_otsu_thresh = QtGui.QImage(otsu_thresh.data, width, height, bytes_per_line,QtGui.QImage.Format_Grayscale8)
            self.label3.setPixmap(QtGui.QPixmap.fromImage(q_otsu_thresh))
            self.label3.setScaledContents(True)

            # Display sure_bg result in label4
            height, width = sure_bg.shape
            bytes_per_line = width
            q_img_sure_bg = QtGui.QImage(sure_bg.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            self.label4.setPixmap(QtGui.QPixmap.fromImage(q_img_sure_bg))
            self.label4.setScaledContents(True)

            # Display sure_fg result in label5
            q_img_sure_fg = QtGui.QImage(sure_fg.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            self.label5.setPixmap(QtGui.QPixmap.fromImage(q_img_sure_fg))
            self.label5.setScaledContents(True)

            # Display markers result in label6
            markers_img = np.zeros_like(markers, dtype=np.uint8)
            markers_img[markers == -1] = 255  # Boundary markers
            q_img_markers = QtGui.QImage(markers_img.data, width, height, bytes_per_line,
                                         QtGui.QImage.Format_Grayscale8)
            self.label6.setPixmap(QtGui.QPixmap.fromImage(q_img_markers))
            self.label6.setScaledContents(True)

            # Display the final watershed result in label7
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img_watershed = QtGui.QImage(img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.label7.setPixmap(QtGui.QPixmap.fromImage(q_img_watershed))
            self.label7.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ShowImage()
    window.show()
    sys.exit(app.exec_())
