"""Args: None
Returns: None
Raises: None
Takes UI from Calculator.py and applies functionality to each button.
"""
# Author: Jacob Hallberg
# Last Edited: 12/30/2017
from math import sqrt
from PyQt5 import QtWidgets
from calculator_UI import Ui_Calculator


class Calculator(QtWidgets.QMainWindow, Ui_Calculator):
    """Args: QtWidgets, QMainWindow, Ui_Calculator
    Returns: None
    Raises: None
    """

    def __init__(self, parent=None):
        """Args: None
        Returns: None
        Raises: None
        Initializes all of the needed vars, as well as button click
        functionality.
        """
        super(Calculator, self).__init__(parent)
        self.setupUi(self)

        self.display_string = ""
        self.display_string2 = ""
        self.which_string = 0
        self.repeat = 0
        self.operator = ""

        self.Zero.clicked.connect(self.button_clicked)
        self.One.clicked.connect(self.button_clicked)
        self.Two.clicked.connect(self.button_clicked)
        self.Three.clicked.connect(self.button_clicked)
        self.Four.clicked.connect(self.button_clicked)
        self.Five.clicked.connect(self.button_clicked)
        self.Six.clicked.connect(self.button_clicked)
        self.Seven.clicked.connect(self.button_clicked)
        self.Eight.clicked.connect(self.button_clicked)
        self.Nine.clicked.connect(self.button_clicked)
        self.Minus.clicked.connect(self.button_clicked)
        self.Plus.clicked.connect(self.button_clicked)
        self.Divide.clicked.connect(self.button_clicked)
        self.Multiply.clicked.connect(self.button_clicked)
        self.Equal.clicked.connect(self.button_clicked)
        self.Decimal.clicked.connect(self.button_clicked)
        self.PlusMinus.clicked.connect(self.button_clicked)
        self.Sqrt.clicked.connect(self.button_clicked)
        self.C.clicked.connect(self.button_clicked)
        self.Back.clicked.connect(self.button_clicked)

    def button_clicked(self):
        """Args: None
        Returns: None
        Raises: None
        All button clicks from the user are sent to this function
        and then routed to other helper functions
        """
        sender = self.sender()
        if sender.text() in {'+', '-', '÷', '*', '='}:
            self.operations()
        elif sender.text() == 'C':
            self.clear()
        elif sender.text() == '←':
            self.back_space()
        elif sender.text() == '√':
            self.s_root()
        elif sender.text() == '.':
            self.add_decimal()
        elif sender.text() == '±':
            self.add_remove_minus()
        else:
            self.string_appender()

    def operations(self):
        """Args: None
        Returns: None
        Raises: None
        Views and makes decisions based on the sent string from
        each operational button click.
        """

        sender = self.sender()
        self.repeat = 0
        self.which_string = 1
        if sender.text() == '+':
            self.operator = '+'
        if sender.text() == '-':
            self.operator = '-'
        if sender.text() == '÷':
            self.operator = '÷'
        if sender.text() == '*':
            self.operator = '*'
        if sender.text() == '=':
            if self.display_string2 == "":
                self.clear()
            else:    
                if self.operator == '+':
                    self.display_string = str(
                        float(self.display_string) + float(self.display_string2))
                if self.operator == '-':
                    self.display_string = str(
                        float(self.display_string) - float(self.display_string2))
                if self.operator == '÷':
                    if self.display_string2 == '0':
                        self.clear()
                    else: 
                        self.display_string = str(
                        format(float(self.display_string) / float(self.display_string2), '.2f'))
                if self.operator == '*':
                    self.display_string = str(
                        float(self.display_string) * float(self.display_string2))

                self.display_string2 = ""
                self.which_string = 0
                self.repeat = 1
                self.NumberField.display(self.display_string)

    def string_appender(self):
        """Args: None
        Returns: None
        Raises: None
        Each non-operational button press is treated as an append to a string
        using string concatination.
        """
        sender = self.sender()
        if self.which_string == 0 and self.repeat == 0:
            if len(self.display_string) >= 16:
                # Need to add error msg here.
                print("Calculator limited to 16 digits.")
            else:    
                self.display_string = self.display_string + sender.text()
                self.NumberField.display(self.display_string)
        elif self.which_string == 1 and self.repeat == 0:
            if len(self.display_string2) >= 16:
                # Need to add error msg here.
                print("Calculator limited to 16 digits.")
            else:    
                self.display_string2 = self.display_string2 + sender.text()
                self.NumberField.display(self.display_string2)
        else:
            self.display_string = "" + sender.text()
            self.repeat = 0
            self.NumberField.display(self.display_string)

    def add_remove_minus(self):
        """Args: None
        Returns: None
        Raises: None
        Checks if current string has a minus in the front, if it does
        the minus is removed. Otherwise, a minus is added.
        """
        if self.which_string == 0:
            if '-' in self.display_string:
                self.display_string = self.display_string[1:]
                self.NumberField.display(self.display_string)
            else:
                self.display_string = '-' + self.display_string
                self.NumberField.display(self.display_string)
        if self.which_string == 1:
            if '-' in self.display_string2:
                self.display_string2 = self.display_string2[1:]
                self.NumberField.display(self.display_string2)
            else:
                self.display_string2 = '-' + self.display_string2
                self.NumberField.display(self.display_string2)

    def add_decimal(self):
        """Args: None
        Returns: None
        Raises: None
        Uses string concatination to add a decimals to the end of a string.
        """
        sender = self.sender()
        if self.which_string == 0:
            if '.' in self.display_string:
                print("Uable to add Decimal")
            else:
                self.display_string = self.display_string + sender.text()
                self.NumberField.display(self.display_string)
        else:
            if '.' in self.display_string2:
                print("Uable to add Decimal")
            else:
                self.display_string2 = self.display_string2 + sender.text()
                self.NumberField.display(self.display_string2)

    def back_space(self):
        """Args: None
        Returns: None
        Raises: None
        Removes the last character of the current string.
        """
        if self.which_string == 0:
            if len(self.display_string) == 1:
                self.display_string = ""
            else:
                self.display_string = self.display_string[:-1]
            self.NumberField.display(self.display_string)
        if self.which_string == 1:
            if len(self.display_string2) == 1:
                self.display_string2 = ""
            else:
                self.display_string2 = self.display_string[:-1]
            self.NumberField.display(self.display_string2)

    def s_root(self):
        """Args: None
        Returns: None
        Raises: None
        Performs a square root on current string.
        """
        if self.which_string == 0:
            self.display_string = str(
                format(sqrt(float(self.display_string)), '.2f'))
            self.NumberField.display(self.display_string)
        if self.which_string == 1:
            self.display_string2 = str(
                format(sqrt(float(self.display_string2)), '.2f'))
            self.NumberField.display(self.display_string2)

    def clear(self):
        """Args: None
        Returns: None
        Raises: None
        Resets all variables to their default state. Clears calculator.
        """
        self.display_string = ""
        self.display_string2 = ""
        self.which_string = 0
        self.repeat = 0
        self.operator = ""
        self.NumberField.display('0')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    nextGui = Calculator()
    nextGui.show()
    sys.exit(app.exec_())