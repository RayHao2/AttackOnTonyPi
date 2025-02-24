#!/usr/bin/env python3
# encoding: utf-8
import os
import re
import cv2
import sys
import copy
import math
import time
import threading
from socket import * 
from ui import Ui_Form
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from servo_controller import *
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtSql import QSqlDatabase, QSqlQuery

class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.set_window_position()
        
        self.path = os.path.split(os.path.realpath(__file__))[0]
        self.setWindowIcon(QIcon(os.path.join(self.path, 'resources/app.png')))
        self.message = QMessageBox()

        self.radioButton_zn.toggled.connect(lambda: self.language(self.radioButton_zn))
        self.radioButton_en.toggled.connect(lambda: self.language(self.radioButton_en))        
        self.radioButton_zn.setChecked(True)
        # 根据系统语言进行切换
        result1 = os.popen('locale |grep LC_ALL').read().replace('\n', '').split('=')[1]
        result2 = os.popen('locale |grep LANG').read().replace('\n', '').split('=')[1]
        if result1 == '':
            if 'CN' in result2:
                self.chinese = True
                self.radioButton_zn.setChecked(True)
            else:
                self.chinese = False
                self.radioButton_en.setChecked(True)
        else:
            if 'CN' in result1:
                self.chinese = True
                self.radioButton_zn.setChecked(True)
            else:
                self.chinese = False
                self.radioButton_en.setChecked(True)  
        #################################界面#######################################
        self.id = 0
        self.dev = 0
        self.servoTemp = 0
        self.servoMin = 0
        self.servoMax = 0
        self.servoMinV = 0
        self.servoMaxV = 0
        self.servoMove = 0
        self.horizontalSlider_servoTemp.valueChanged.connect(lambda: self.horizontalSlider_valuechange('servoTemp'))
        self.horizontalSlider_servoMin.valueChanged.connect(lambda: self.horizontalSlider_valuechange('servoMin'))
        self.horizontalSlider_servoMax.valueChanged.connect(lambda: self.horizontalSlider_valuechange('servoMax'))
        self.horizontalSlider_servoMinV.valueChanged.connect(lambda: self.horizontalSlider_valuechange('servoMinV'))
        self.horizontalSlider_servoMaxV.valueChanged.connect(lambda: self.horizontalSlider_valuechange('servoMaxV'))
        self.horizontalSlider_servoMove.valueChanged.connect(lambda: self.horizontalSlider_valuechange('servoMove'))

        self.pushButton_read.pressed.connect(lambda: self.button_clicked('read'))
        self.pushButton_set.pressed.connect(lambda: self.button_clicked('set'))
        self.pushButton_default.pressed.connect(lambda: self.button_clicked('default'))
        self.pushButton_quit2.pressed.connect(lambda: self.button_clicked('quit2'))
        self.pushButton_resetPos.pressed.connect(lambda: self.button_clicked('resetPos'))
        self.lineEdit_servoDev.setValidator(QIntValidator(-125, 125))
        
        self.readOrNot = False
        self.haved_read = False

    def set_window_position(self):
        # 窗口居中
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def language(self, name):
        if self.radioButton_zn.isChecked() and name.text() == '中文':
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), "舵机调试工具")
            self.chinese = True
            self.label_id.setText("舵机ID")
            self.label_dev.setText("舵机偏差")
            self.label_temp.setText("温度范围")
            self.label_angle.setText("角度范围")
            self.label_vin.setText("电压范围")
            self.label_current_angle.setText("当前角度: ")
            self.label_current_temp.setText("当前温度: ")
            self.label_current_vin.setText("当前电压: ")
            self.label_reset.setText("舵机调试")
            self.pushButton_resetPos.setText("中位")
            self.pushButton_quit2.setText("退出")
            self.label_tips.setText("<html><head/><body><p><span style=\" color:#ff0000;\">注意：使用下面的功能时，请确保控制器只连接了一个舵机，否则会引起冲突!</span></p></body></html>")
            self.pushButton_read.setText("读取")
            self.pushButton_set.setText("设置")
            self.pushButton_default.setText("默认")
            self.message_From('使用此面板时，请确保只连接了一个舵机，否则会引起冲突！')
        elif self.radioButton_en.isChecked() and name.text() == 'English':
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), "Servo Tool")
            self.chinese = False
            self.label_id.setText("Servo ID")
            self.label_dev.setText("Servo deviation")
            self.label_temp.setText("Temperature")
            self.label_angle.setText("Angle")
            self.label_vin.setText("Voltage")
            self.label_current_angle.setText("Current angle: ")
            self.label_current_temp.setText("Current temperature: ")
            self.label_current_vin.setText("Current voltage: ")
            self.label_reset.setText("Servo Test")
            self.pushButton_resetPos.setText("Middle position")
            self.pushButton_quit2.setText("Quit")
            self.label_tips.setText("<html><head/><body><p><span style=\" color:#ff0000;\">Note:before debugging servo,make sure that the servo controller is connected with ONE servo.Otherwise it may cause a conflict!</span></p></body></html>")
            self.pushButton_read.setText("Read")
            self.pushButton_set.setText("Setting")
            self.pushButton_default.setText("Default")
            self.message_From('Before debugging servo,make sure that the servo controller is connected with ONE servo.Otherwise it may cause a conflict!')

    # 弹窗提示函数
    def message_from(self, str):
        try:
            QMessageBox.about(self, '', str)
            time.sleep(0.01)
        except:
            pass
    
    def message_From(self, str):
        self.message_from(str)
   
    # 弹窗提示函数
    def message_delect(self, str):
        messageBox = QMessageBox()
        messageBox.setWindowTitle(' ')
        messageBox.setText(str)
        messageBox.addButton(QPushButton('OK'), QMessageBox.YesRole)
        messageBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)
        return messageBox.exec_()

    # 窗口退出
    def closeEvent(self, e):        
        result = QMessageBox.question(self,
                                    "关闭窗口提醒",
                                    "exit?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        if result == QMessageBox.Yes:
            QWidget.closeEvent(self, e)
        else:
            e.ignore()
    
    def horizontalSlider_valuechange(self, name):
        if name == 'servoTemp':
            self.temp = str(self.horizontalSlider_servoTemp.value())
            self.label_servoTemp.setText(self.temp + '℃')
        if name == 'servoMin':
            self.servoMin = str(self.horizontalSlider_servoMin.value())
            self.label_servoMin.setText(self.servoMin)
        if name == 'servoMax':
            self.servoMax = str(self.horizontalSlider_servoMax.value())
            self.label_servoMax.setText(self.servoMax)
        if name == 'servoMinV':
            self.servoMinV = str(self.horizontalSlider_servoMinV.value()/10)
            self.label_servoMinV.setText(self.servoMinV + 'V')
        if name == 'servoMaxV':
            self.servoMaxV = str(self.horizontalSlider_servoMaxV.value()/10)
            self.label_servoMaxV.setText(self.servoMaxV + 'V')
        if name == 'servoMove':
            self.servoMove = str(self.horizontalSlider_servoMove.value())            
            self.label_servoMove.setText(self.servoMove)
            setServoPulse(self.id, int(self.servoMove), 0)
    
    def button_clicked(self, name):
        if name == 'read':
            try:
                self.id = getBusServoID()
                if self.id is None:
                    if self.chinese:
                        self.message_From('读取id失败')
                    else:
                        self.message_From('Failed to read ID')
                    return
                self.readOrNot = True
                self.dev = getServoDeviation(self.id)
                if self.dev > 125:
                    self.dev = -(0xff-(self.dev - 1))
                self.servoTemp = getBusServoTempLimit(self.id)
                (self.servoMin, self.servoMax) = getBusServoAngleLimit(self.id)
                (self.servoMinV, self.servoMaxV) = getBusServoVinLimit(self.id)
                self.servoMove = getServoPulse(self.id)
                
                currentVin = getBusServoVin(self.id)

                currentTemp = getBusServoTemp(self.id)

                self.lineEdit_servoID.setText(str(self.id))
                self.lineEdit_servoDev.setText(str(self.dev))
                
                self.horizontalSlider_servoTemp.setValue(self.servoTemp)
                self.horizontalSlider_servoMin.setValue(self.servoMin)
                self.horizontalSlider_servoMax.setValue(self.servoMax)
                MinV = self.servoMinV
                MaxV = self.servoMaxV            
                self.horizontalSlider_servoMinV.setValue(int(MinV/100))
                self.horizontalSlider_servoMaxV.setValue(int(MaxV/100))

                self.label_servoCurrentP.setText(str(self.servoMove))
                self.label_servoCurrentV.setText(str(round(currentVin/1000.0, 2)) + 'V')
                self.label_servoCurrentTemp.setText(str(currentTemp) + '℃')

                self.horizontalSlider_servoMove.setValue(self.servoMove)
            except:
                if self.chinese:
                    self.message_From('读取超时')
                else:
                    self.message_From('Read timeout')
                return
            if self.chinese:
                self.message_From('读取成功')
            else:
                self.message_From('Success')
        if name == 'set':
            if self.readOrNot is False:
                if self.chinese:
                    self.message_From('请先读取，否则无法获取舵机信息，从而进行设置！')
                else:
                    self.message_From('Read first！')
                return
            if self.chinese:
                if self.message_delect('此操作会对所有连接的舵机进行设置，请确保只连接了需要设置的舵机，是否继续？'):
                    return 
            else:
                if self.message_delect('This operation will set all the connected servos, please make sure whether the servo that needs to be set up is connected, whether to continue'):
                    return 
            self.readOrNot = False
            ID = self.lineEdit_servoID.text()
            if ID == '':
                if self.chinese:
                    self.message_From('舵机id参数为空，无法设置')
                else:
                    self.message_From('Please input id')
                return           
            dev = self.lineEdit_servoDev.text()
            if dev is '':
                dev = 0
            dev = int(dev)
            if dev > 125 or dev < -125:
                if self.chinese:
                    self.message_From('偏差参数超出可调节范围-125～125，无法设置')
                else:
                    self.message_From('Deviation out of range -125~125')
                return          
            temp = self.horizontalSlider_servoTemp.value()
            pos_min = self.horizontalSlider_servoMin.value()
            pos_max = self.horizontalSlider_servoMax.value()
            if pos_min > pos_max:
                if self.chinese:
                    self.message_From('舵机范围参数错误，无法设置')
                else:
                    self.message_From('Wrong angle range')
                return
            vin_min = self.horizontalSlider_servoMinV.value()
            vin_max = self.horizontalSlider_servoMaxV.value()
            if vin_min > vin_max:
                if self.chinese:
                    self.message_From('舵机电压范围参数错误，无法设置')
                else:
                    self.message_From('Wrong voltage range')
                return
            pos = self.horizontalSlider_servoMove.value()
            
            ID = int(ID)
            
            try:
                setBusServoID(self.id, ID)
                time.sleep(0.01)
                if getBusServoID() != ID:
                    if self.chinese:
                        self.message_From('id设置失败！')
                    else:
                        self.message_From('Failed！')
                    return
                setBusServoDeviation(ID, dev)
                time.sleep(0.01)
                saveServoDeviation(ID)
                time.sleep(0.01)
                d = getServoDeviation(ID)
                if d > 125:
                    d = -(0xff-(d - 1))               
                if d != dev:
                    if self.chinese:
                        self.message_From('偏差设置失败！')
                    else:
                        self.message_From('Failed！')
                    return            
                setBusServoMaxTemp(ID, temp)
                time.sleep(0.01)
                if getBusServoTempLimit(ID) != temp:
                    if self.chinese:
                        self.message_From('温度设置失败！')
                    else:
                        self.message_From('Failed！')

                    return 
                setBusServoAngleLimit(ID, pos_min, pos_max)
                time.sleep(0.01)
                if getBusServoAngleLimit(ID) != (pos_min, pos_max):
                    if self.chinese:
                        self.message_From('角度范围设置失败！')
                    else:
                        self.message_From('Failed！')
                    return 
                setBusServoVinLimit(ID, vin_min*100, vin_max*100)
                time.sleep(0.01)
                if getBusServoVinLimit(ID) != (vin_min*100, vin_max*100):
                    if self.chinese:
                        self.message_From('电压范围设置失败！')
                    else:
                        self.message_From('Failed！')
                    return 
                setServoPulse(ID, pos, 0)
            except:
                if self.chinese:
                    self.message_From('设置超时!')
                else:
                    self.message_From('Timeout!')
                return                
            if self.chinese: 
                self.message_From('设置成功')
            else:
                self.message_From('success')

        if name == 'default':
            if self.readOrNot is False:
                if self.chinese:
                    self.message_From('请先读取，否则无法获取舵机信息，从而进行设置！')
                else:
                    self.message_From('Read first！')
                return
            if self.chinese:
                if self.message_delect('此操作会对所有连接的舵机进行设置，请确保只连接了需要设置的舵机，是否继续？'):
                    return 
            else:
                if self.message_delect('This operation will set all the connected servos, please make sure whether the servo that needs to be set up is connected, whether to continue'):
                    return 
            self.readOrNot = False
            try:
                setBusServoID(self.id, 1)
                time.sleep(0.01)
                if getBusServoID() != 1:
                    if self.chinese:
                        self.message_From('id设置失败！')
                    else:
                        self.message_From('Failed！')
                    return
                setBusServoDeviation(1, 0)
                time.sleep(0.01)
                saveServoDeviation(1)
                time.sleep(0.01)
                if getServoDeviation(1) != 0:
                    if self.chinese:
                        self.message_From('偏差设置失败！')
                    else:
                        self.message_From('Failed！')
                    return
                setBusServoMaxTemp(1, 85)
                time.sleep(0.01)
                if getBusServoTempLimit(1) != 85:
                    if self.chinese:
                        self.message_From('温度设置失败！')
                    else:
                        self.message_From('Failed！')
                    return
                setBusServoAngleLimit(1, 0, 1000)
                time.sleep(0.01)
                if getBusServoAngleLimit(1) != (0, 1000):
                    if self.chinese:
                        self.message_From('角度范围设置失败！')
                    else:
                        self.message_From('Failed！')
                    return          
                setBusServoVinLimit(1, 4500, 14000)
                time.sleep(0.01)
                if getBusServoVinLimit(1) != (4500, 14000):
                    if self.chinese:
                        self.message_From('电压范围设置失败！')
                    else:
                        self.message_From('Failed！')
                    return             
                setServoPulse(1, 500, 0)
            except:
                if self.chinese:
                    self.message_From('设置超时!')
                else:
                    self.message_From('Timeout!')
                return
            if self.chinese:
                self.message_From('设置成功')
            else:
                self.message_From('Success')
        if name == 'quit2':
            self.camera_ui = True
            self.camera_ui_break = True
            try:
                self.cap.release()
            except:
                pass          
            sys.exit()
        if name == 'resetPos':
            self.horizontalSlider_servoMove.setValue(500)
            setServoPulse(self.id, 500, 0)

if __name__ == "__main__":  
    app = QtWidgets.QApplication(sys.argv)
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())
