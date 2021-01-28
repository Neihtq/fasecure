"""
Generated python code using PyQt5 Designer. Need to fix bad formatting
as this is generated UI code after making the UI using PyQt5 Designer.
"""
# Author: Jacob Hallberg
# Last Edited: 12/30/2017


from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Calculator(object):
    def setupUi(self, Calculator):
        Calculator.setObjectName("Calculator")
        Calculator.setEnabled(True)
        Calculator.resize(450, 681)

        self.Five = QtWidgets.QPushButton(Calculator)
        self.Five.setGeometry(QtCore.QRect(120, 380, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Five.setFont(font)
        self.Five.setObjectName("Five")

        self.Six = QtWidgets.QPushButton(Calculator)
        self.Six.setGeometry(QtCore.QRect(230, 380, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Six.setFont(font)
        self.Six.setObjectName("Six")

        self.Seven = QtWidgets.QPushButton(Calculator)
        self.Seven.setGeometry(QtCore.QRect(10, 280, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Seven.setFont(font)
        self.Seven.setObjectName("Seven")

        self.Sqrt = QtWidgets.QPushButton(Calculator)
        self.Sqrt.setGeometry(QtCore.QRect(120, 180, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Sqrt.setFont(font)
        self.Sqrt.setObjectName("Sqrt")

        self.Multiply = QtWidgets.QPushButton(Calculator)
        self.Multiply.setGeometry(QtCore.QRect(340, 380, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Multiply.setFont(font)
        self.Multiply.setObjectName("Multiply")

        self.Decimal = QtWidgets.QPushButton(Calculator)
        self.Decimal.setGeometry(QtCore.QRect(120, 580, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Decimal.setFont(font)
        self.Decimal.setObjectName("Decimal")

        self.Eight = QtWidgets.QPushButton(Calculator)
        self.Eight.setGeometry(QtCore.QRect(120, 280, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Eight.setFont(font)
        self.Eight.setObjectName("Eight")

        self.Divide = QtWidgets.QPushButton(Calculator)
        self.Divide.setGeometry(QtCore.QRect(340, 280, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Divide.setFont(font)
        self.Divide.setObjectName("Divide")

        self.PlusMinus = QtWidgets.QPushButton(Calculator)
        self.PlusMinus.setGeometry(QtCore.QRect(230, 180, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.PlusMinus.setFont(font)
        self.PlusMinus.setObjectName("PlusMinus")

        self.Equal = QtWidgets.QPushButton(Calculator)
        self.Equal.setGeometry(QtCore.QRect(230, 580, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Equal.setFont(font)
        self.Equal.setObjectName("Equal")

        self.Zero = QtWidgets.QPushButton(Calculator)
        self.Zero.setGeometry(QtCore.QRect(10, 580, 101, 91))
        
        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Zero.setFont(font)
        self.Zero.setObjectName("Zero")

        self.C = QtWidgets.QPushButton(Calculator)
        self.C.setEnabled(True)
        self.C.setGeometry(QtCore.QRect(340, 180, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.C.setFont(font)
        self.C.setObjectName("C")

        self.Nine = QtWidgets.QPushButton(Calculator)
        self.Nine.setGeometry(QtCore.QRect(230, 280, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Nine.setFont(font)
        self.Nine.setObjectName("Nine")

        self.Minus = QtWidgets.QPushButton(Calculator)
        self.Minus.setGeometry(QtCore.QRect(340, 480, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Minus.setFont(font)
        self.Minus.setCheckable(False)
        self.Minus.setObjectName("Minus")

        self.Three = QtWidgets.QPushButton(Calculator)
        self.Three.setGeometry(QtCore.QRect(230, 480, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Three.setFont(font)
        self.Three.setObjectName("Three")

        self.One = QtWidgets.QPushButton(Calculator)
        self.One.setGeometry(QtCore.QRect(10, 480, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.One.setFont(font)
        self.One.setObjectName("One")

        self.Four = QtWidgets.QPushButton(Calculator)
        self.Four.setGeometry(QtCore.QRect(10, 380, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Four.setFont(font)
        self.Four.setObjectName("Four")

        self.NumberField = QtWidgets.QLCDNumber(Calculator)
        self.NumberField.setGeometry(QtCore.QRect(13, 10, 421, 161))
        self.NumberField.setSmallDecimalPoint(False)
        self.NumberField.setDigitCount(16)
        self.NumberField.setObjectName("NumberField")

        self.Plus = QtWidgets.QPushButton(Calculator)
        self.Plus.setGeometry(QtCore.QRect(340, 580, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Plus.setFont(font)
        self.Plus.setObjectName("Plus")

        self.Two = QtWidgets.QPushButton(Calculator)
        self.Two.setGeometry(QtCore.QRect(120, 480, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(18)

        self.Two.setFont(font)
        self.Two.setObjectName("Two")

        self.Back = QtWidgets.QPushButton(Calculator)
        self.Back.setGeometry(QtCore.QRect(10, 180, 101, 91))

        font = QtGui.QFont()
        font.setFamily("STIX")
        font.setPointSize(36)
        self.Back.setFont(font)
        
        self.Back.setObjectName("Back")

        self.retranslateUi(Calculator)
        QtCore.QMetaObject.connectSlotsByName(Calculator)

    def retranslateUi(self, Calculator):
        _translate = QtCore.QCoreApplication.translate
        Calculator.setWindowTitle(_translate("Calculator", "Jacob Hallberg\'s Calculator"))
        self.Five.setText(_translate("Calculator", "5"))
        self.Six.setText(_translate("Calculator", "6"))
        self.Seven.setText(_translate("Calculator", "7"))
        self.Sqrt.setText(_translate("Calculator", "√"))
        self.Multiply.setText(_translate("Calculator", "*"))
        self.Decimal.setText(_translate("Calculator", "."))
        self.Eight.setText(_translate("Calculator", "8"))
        self.Divide.setText(_translate("Calculator", "÷"))
        self.PlusMinus.setText(_translate("Calculator", "±"))
        self.Equal.setText(_translate("Calculator", "="))
        self.Zero.setText(_translate("Calculator", "0"))
        self.C.setText(_translate("Calculator", "C"))
        self.Nine.setText(_translate("Calculator", "9"))
        self.Minus.setText(_translate("Calculator", "-"))
        self.Three.setText(_translate("Calculator", "3"))
        self.One.setText(_translate("Calculator", "1"))
        self.Four.setText(_translate("Calculator", "4"))
        self.Plus.setText(_translate("Calculator", "+"))
        self.Two.setText(_translate("Calculator", "2"))
        self.Back.setText(_translate("Calculator", "←"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Calculator = QtWidgets.QDialog()
    ui = Ui_Calculator()
    ui.setupUi(Calculator)
    Calculator.show()
    sys.exit(app.exec_())