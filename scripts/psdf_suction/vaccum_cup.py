import serial
import time

class VaccumCup():
    def __init__(self, port="/dev/ttyUSB0"):
        try:
            self.s = serial.Serial(port, baudrate=9600)
            self.release()
            time.sleep(3)
        except Exception as e:
            print(e)
        print("serial port state", self.s.isOpen())

    # def release(self):
    #     self.s.write("OFF".encode('utf-8'))
    #     #print(self.s.write(b"OFF"))
    #
    # def grasp(self):
    #     self.s.write("ON".encode('utf-8'))
    #     #print(self.s.write(b"ON"))

    def release(self):
        self.s.write("0".encode('utf-8'))
        #print(self.s.write(b"OFF"))

    def ur5_grasp(self):
        self.s.write("1".encode('utf-8'))
        #print(self.s.write(b"ON"))
    def ur5_touch(self):
        self.s.write("2".encode('utf-8'))
    
    def ur5_release(self):
        self.s.write("3".encode('utf-8'))
    
    def ur10_grasp(self):
        self.s.write("5".encode('utf-8'))
    
    def ur10_touch(self):
        self.s.write("6".encode('utf-8'))
    
    def ur10_release(self):
        self.s.write("7".encode('utf-8'))

