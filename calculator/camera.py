#Importing necessary libraries, mainly the OpenCV, and PyQt libraries
import cv2
import numpy as np
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal


from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

class ShowVideo(QtCore.QObject):

	#initiating the built in camera
	camera_port = 0
	camera = cv2.VideoCapture(camera_port)
	VideoSignal = QtCore.pyqtSignal(QtGui.QImage)

	def __init__(self, parent = None):
		super(ShowVideo, self).__init__(parent)

	@QtCore.pyqtSlot()
	def startVideo(self):

		run_video = True
		while run_video:
			ret, image = self.camera.read()
			
			color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			height, width, _ = color_swapped_image.shape
			
			qt_image = QtGui.QImage(color_swapped_image.data,
									width, 
									height,
			  						color_swapped_image.strides[0],
			   						QtGui.QImage.Format_RGB888)

			pixmap = QtGui.QPixmap(qt_image)
			qt_image = pixmap.scaled(1000, 800, QtCore.Qt.KeepAspectRatio)
			qt_image = QtGui.QImage(qt_image)

			self.VideoSignal.emit(qt_image)



class ImageViewer(QtWidgets.QWidget):
	def __init__(self, parent = None):
		super(ImageViewer, self).__init__(parent)
		self.image = QtGui.QImage()
		self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)



	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		painter.drawImage(0,0, self.image)
		self.image = QtGui.QImage()

	def initUI(self):
		self.setWindowTitle('Test')

	
	@QtCore.pyqtSlot(QtGui.QImage)
	def setImage(self, image):
		if image.isNull():
			print("viewer dropped frame!")

		self.image = image
		if image.size() != self.size():
			self.setFixedSize(image.size())
		self.update()

def button1_clicked():
   print("Button 1 clicked")

def button2_clicked():
   print("Button 2 clicked")   
   
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    thread = QtCore.QThread()
    thread.start()

    vid = ShowVideo()
    vid.moveToThread(thread)
    image_viewer = ImageViewer()
    #image_viewer.resize(200,400)


    vid.VideoSignal.connect(image_viewer.setImage)

    #Button to start the videocapture:

    push_button = QtWidgets.QPushButton('Start')
    push_button.clicked.connect(vid.startVideo)
    vertical_layout = QtWidgets.QVBoxLayout()

    vertical_layout.addWidget(image_viewer)
    vertical_layout.addWidget(push_button)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    #app = QApplication(sys.argv)
    widget = QWidget()

    button1 = QPushButton(widget)
    button1.setText("Button1")
    button1.move(2000,32)
    button1.clicked.connect(button1_clicked)

    button2 = QPushButton(widget)
    button2.setText("Button2")
    button2.move(2500,64)
    button2.clicked.connect(button2_clicked)

    widget.setGeometry(50,50,320,200)
    widget.setWindowTitle("PyQt5 Button Click Example")
    #widget.show()
    #sys.exit(app.exec_())


    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.resize(3000,800)
    main_window.show()
    sys.exit(app.exec_())



