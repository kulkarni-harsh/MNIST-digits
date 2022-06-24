import sys
from typing import final
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
from keras.models import load_model


class MainWindow(QWidget):
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(270, 270)
        self.setWindowTitle('Painter Board')
        self.tracing_xy = []
        self.lineHistory = []
        self.pen = QPen(Qt.black, 10, Qt.SolidLine)
        self.matrix=np.zeros((28,28))
        self.initUI()
        

    def paintEvent(self, QPaintEvent):
        self.painter = QPainter()
        self.painter.begin(self)
        self.painter.setPen(self.pen)

        start_x_temp = 0
        start_y_temp = 0

        if self.lineHistory:
            for line_n in range(len(self.lineHistory)):
                for point_n in range(1, len(self.lineHistory[line_n])):
                    start_x, start_y = self.lineHistory[line_n][point_n-1][0], self.lineHistory[line_n][point_n-1][1]
                    end_x, end_y = self.lineHistory[line_n][point_n][0], self.lineHistory[line_n][point_n][1]
                    self.painter.drawLine(start_x, start_y, end_x, end_y)

        for x, y in self.tracing_xy:
            if start_x_temp == 0 and start_y_temp == 0:
                self.painter.drawLine(self.start_xy[0][0], self.start_xy[0][1], x, y)
            else:
                self.painter.drawLine(start_x_temp, start_y_temp, x, y)
                # print(start_x_temp, start_y_temp, x, y)
                if start_x_temp==x:
                    for j in range(min(start_y_temp,y),max(start_y_temp,y)+1):
                        try:
                            self.matrix[j//10][x//10]=1
                            if x//10>0 and x//10<27 and j//10<27 and j//10>0:
                                self.matrix[j//10-1][x//10+1]=0.5
                                self.matrix[j//10-1][x//10-1]=0.5
                                self.matrix[j//10+1][x//10-1]=0.5
                                self.matrix[j//10+1][x//10+1]=0.5

                                self.matrix[j//10+1][x//10]=0.5
                                self.matrix[j//10-1][x//10]=0.5
                                self.matrix[j//10][x//10+1]=0.5
                                self.matrix[j//10][x//10-1]=0.5
                        finally:
                            continue
                else:
                    for i in range(min(start_x_temp,x),max(start_x_temp,x)+1):
                        try:
                            self.matrix[y//10][i//10]=1
                            if y//10>0 and y//10<27 and i//10<27 and i//10>0:
                                self.matrix[y//10-1][i//10-1]=0.5
                                self.matrix[y//10-1][i//10+1]=0.5
                                self.matrix[y//10+1][i//10-1]=0.5
                                self.matrix[y//10+1][i//10+1]=0.5

                                self.matrix[y//10+1][i//10]=0.5
                                self.matrix[y//10-1][i//10]=0.5
                                self.matrix[y//10][i//10+1]=0.5
                                self.matrix[y//10][i//10-1]=0.5
                        finally:
                            continue


            start_x_temp = x
            start_y_temp = y

        self.painter.end()

    def mousePressEvent(self, QMouseEvent):
        try:
            self.matrix[QMouseEvent.pos().y()//10][QMouseEvent.pos().x()//10]=1
        finally:
            self.start_xy = [(QMouseEvent.pos().x(), QMouseEvent.pos().y())]
        

    def mouseMoveEvent(self, QMouseEvent):
        try:
            self.matrix[QMouseEvent.pos().y()//10][QMouseEvent.pos().x()//10]=1
        finally:
            self.tracing_xy.append((QMouseEvent.pos().x(), QMouseEvent.pos().y()))
        
        self.update()

    def mouseReleaseEvent(self, QMouseEvent):
        try:
            self.matrix[QMouseEvent.pos().y()//10][QMouseEvent.pos().x()//10]=1
        finally:
            self.tracing_xy = []
            self.lineHistory.append(self.start_xy+self.tracing_xy)

    def initUI(self):
        self.setGeometry(200, 200, 270, 270)
        self.setWindowTitle("Number Recognition")

        self.b1 = QPushButton(self)
        self.b1.move(180,240)
        self.b1.setText("Recognise!")
        self.b1.clicked.connect(self.predict_button_clicked)

        self.b2 = QPushButton(self)
        self.b2.move(10,240)
        self.b2.setText("Clear")
        self.b2.clicked.connect(self.clearCanvas)

    def show_info_messagebox(self,number):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"The number is recognized to be {number}")
        msg.setWindowTitle("Recognition Done")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = msg.exec_()

    def predict_button_clicked(self):
        x=self.matrix.reshape(1,784)
        model=load_model('mnist_model.h5')
        y=model.predict(x,verbose=0)
        y_value=np.argmax(y)
        self.show_info_messagebox(y_value)

    def clearCanvas(self):
        self.tracing_xy = []
        self.lineHistory = []
        self.matrix=np.zeros((28,28))
        self.update()



if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())