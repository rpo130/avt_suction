from ctypes import *
from time import sleep

from attr import dataclass

# import USB-CAN module
import ControlCAN


class GinkGo():
    DevType = ControlCAN.VCI_USBCAN2
    """
    DeviceIndex:
        Ginkgo product (for I2C,SPI, CAN adapters, and Sniffers, etc) supply the way one PC could connect multiple products and 
        working simultaneously, the DeviceIndex is used to identify and distinguish those adapters. 
        For example, one PC connected two CAN Interface (adapters), then DeviceIndex = 0 is used to specify first one, and 
        DeviceIndex = 1 is used to specify the second one. This python program is used to test one CAN interface (or adapter), so 
        it's fixed to 0, if have multiple adapters connected to one PC, then please modify this parameter for different devices
    self.CANIndex:
        In one CAN Interface (or adapter), printed on form factor (the shell on top), it has two CAN channels, CAN1 and CAN2, 
        in Extend (recommanded, or classic) CAN software, it could be selected by "CAN1" (Equal: self.CANIndex = 0)
        or "CAN2"(Equal: self.CANIndex = 1), 
        So when self.CANIndex = 0 here, means CAN1 has been chosen
    CAN_MODE_LOOP_BACK: 
        0: normal working mode, used for CAN channel 1 or 2 communication with other CAN devices (sending and receiving data)
        1: loop back mode, do loop back testing without any wiring with other CAN devices, it's easiest way to do self-testing 
        for CAN adapter
    BaudRate:
        0: 33K
        1: 500K
        2: 1000K
    """
    def __init__(self, DeviceIndex, CANIndex, BaudRate=0, CAN_MODE_LOOP_BACK=0, show=False) -> None:
        self.DeviceIndex = DeviceIndex
        self.CANIndex = CANIndex
        self.BaudRate = BaudRate
        self.CAN_MODE_LOOP_BACK = CAN_MODE_LOOP_BACK

        # initialization process
        self._clear_buffer()
        self._scan_device()
        self._open_device()
        self._init_CAN_ex()
        if show:
            self._read_board_info()
        self._set_filter()

    def _scan_device(self):
        nRet = ControlCAN.VCI_ScanDevice(1) # number of device
        if(nRet == 0):
            print("No device connected!")
            exit()
        else:
            print("Have %d device connected!"%nRet)
    
    def _open_device(self):
        nRet = ControlCAN.VCI_OpenDevice(self.DevType, self.DeviceIndex, 0)
        if(nRet == ControlCAN.STATUS_ERR):
            print("Open device failed!")
            exit()
        else:
            print("Open device success!")

    def _init_CAN_ex(self):
        CAN_InitEx = ControlCAN.VCI_INIT_CONFIG_EX()
        CAN_InitEx.CAN_ABOM = 0
        if(self.CAN_MODE_LOOP_BACK == 1):
            CAN_InitEx.CAN_Mode = 1
        else:
            CAN_InitEx.CAN_Mode = 0
            
        # Baud Rate: 500K
        # SJW, BS1, BS2, PreScale, detail in controlcan.py
        if(self.BaudRate == 0):
            CAN_InitEx.CAN_BRP = 109 #12
            CAN_InitEx.CAN_SJW = 1
            CAN_InitEx.CAN_BS1 = 8   #4
            CAN_InitEx.CAN_BS2 = 1 
        else : #if(BR_1000K == 1):
            CAN_InitEx.CAN_BRP = 9 #12
            CAN_InitEx.CAN_SJW = 1
            CAN_InitEx.CAN_BS1 = 2   #4
            CAN_InitEx.CAN_BS2 = 1 
        
        CAN_InitEx.CAN_NART = 0   #text repeadevicet send management: disable text repeat sending
        CAN_InitEx.CAN_RFLM = 0   #FIFO lock management: new text overwrite old
        CAN_InitEx.CAN_TXFP = 1   #send priority management: by order
        CAN_InitEx.CAN_RELAY = 0  #relay feature enable: close relay function
        nRet = ControlCAN.VCI_InitCANEx(self.DevType, self.DeviceIndex, self.CANIndex, byref(CAN_InitEx))
        if(nRet == ControlCAN.STATUS_ERR):
            print("Init device failed!")
            exit()
        else:
            print("Init device success!")
    
    def _clear_buffer(self):
        ControlCAN.VCI_ClearBuffer(self.DevType, self.DeviceIndex, self.CANIndex)

    def _set_filter(self):
        CAN_FilterConfig = ControlCAN.VCI_FILTER_CONFIG()
        CAN_FilterConfig.FilterIndex = 0
        CAN_FilterConfig.Enable = 1        
        CAN_FilterConfig.ExtFrame = 0
        CAN_FilterConfig.FilterMode = 0
        CAN_FilterConfig.ID_IDE = 0
        CAN_FilterConfig.ID_RTR = 0
        CAN_FilterConfig.ID_Std_Ext = 0
        CAN_FilterConfig.MASK_IDE = 0
        CAN_FilterConfig.MASK_RTR = 0
        CAN_FilterConfig.MASK_Std_Ext = 0
        nRet = ControlCAN.VCI_SetFilter(self.DevType, self.DeviceIndex, self.CANIndex, byref(CAN_FilterConfig))
        if(nRet == ControlCAN.STATUS_ERR):
            print("Set filter failed!")
            exit()
        else:
            print("Set filter success!")

    def _start_CAN(self):
        nRet = ControlCAN.VCI_StartCAN(self.DevType, self.DeviceIndex, self.CANIndex)
        if(nRet == ControlCAN.STATUS_ERR):
            print("Start CAN failed!")
            exit()
        else:
            print("Start CAN success!")
    
    def _read_board_info(self):
        CAN_BoardInfo = ControlCAN.VCI_BOARD_INFO_EX()
        nRet = ControlCAN.VCI_ReadBoardInfoEx(self.DeviceIndex, byref(CAN_BoardInfo))
        if(nRet == ControlCAN.STATUS_ERR):
            print("Get board info failed!")
            exit()
        else:
            print("--CAN_BoardInfo.ProductName = %s"%bytes(CAN_BoardInfo.ProductName).decode('ascii'))
            #print("--CAN_BoardInfo.ProductName = %s"%(CAN_BoardInfo.ProductName))
            print("--CAN_BoardInfo.FirmwareVersion = V%d.%d.%d"%(CAN_BoardInfo.FirmwareVersion[1],CAN_BoardInfo.FirmwareVersion[2],CAN_BoardInfo.FirmwareVersion[3]))
            print("--CAN_BoardInfo.HardwareVersion = V%d.%d.%d"%(CAN_BoardInfo.HardwareVersion[1],CAN_BoardInfo.HardwareVersion[2],CAN_BoardInfo.HardwareVersion[3]))
            print("--CAN_BoardInfo.SerialNumber = ")
            for i in range(0, len(CAN_BoardInfo.SerialNumber)):
                print("%02X"%CAN_BoardInfo.SerialNumber[i], end=" ")
            print()

    def register_receive_cb(self, cb):
        ControlCAN.VCI_RegisterReceiveCallback(self.DeviceIndex, ControlCAN.PVCI_RECEIVE_CALLBACK(cb))

    def logout_receive_cb(self):
        ControlCAN.VCI_LogoutReceiveCallback(self.DeviceIndex)

    def start(self):
        self._start_CAN()

    def read_CAN_status(self):
        CAN_Status = ControlCAN.VCI_CAN_STATUS()
        nRet = ControlCAN.VCI_ReadCANStatus(self.DevType, self.DeviceIndex, self.CANIndex, byref(CAN_Status))
        if(nRet == ControlCAN.STATUS_ERR):
            print("Get CAN status failed!")
        else:
            print("Buffer Size : %d"%CAN_Status.BufferSize)
            print("ESR : 0x%08X"%CAN_Status.regESR)
            print("------Error warning flag : %d"%((CAN_Status.regESR>>0)&0x01))
            print("------Error passive flag : %d"%((CAN_Status.regESR >> 1) & 0x01))
            print("------Bus-off flag : %d"%((CAN_Status.regESR >> 2) & 0x01))
            print("------Last error code(%d) : "%((CAN_Status.regESR>>4)&0x07))
            Error = ["No Error","Stuff Error","Form Error","Acknowledgment Error","Bit recessive Error","Bit dominant Error","CRC Error","Set by software"]
            print(Error[(CAN_Status.regESR>>4)&0x07])

    def send(self):
        CAN_SendData = (ControlCAN.VCI_CAN_OBJ*2)()
        for j in range(0,2):
            CAN_SendData[j].DataLen = 8
            for i in range(0,CAN_SendData[j].DataLen):
                CAN_SendData[j].Data[i] = i+j
            CAN_SendData[j].ExternFlag = 0
            CAN_SendData[j].RemoteFlag = 0
            CAN_SendData[j].ID = 0x155+j
            if(self.CAN_MODE_LOOP_BACK == 1):
                CAN_SendData[j].SendType = 2
            else:
                CAN_SendData[j].SendType = 0
        
        SEND_FRAME_COUNT = 2
        for j in range(0, SEND_FRAME_COUNT):
            nRet = ControlCAN.VCI_Transmit(self.DevType, self.DeviceIndex, self.CANIndex,byref(CAN_SendData), SEND_FRAME_COUNT)
            if(nRet == ControlCAN.STATUS_ERR):
                print("Send CAN data failed!")
            else:
                print("Send CAN data success!")
            sleep(0.1)        

    def receive(self, num):
        self._clear_buffer()
        DataNum = ControlCAN.VCI_GetReceiveNum(self.DevType, self.DeviceIndex, self.CANIndex)
        while DataNum < num:
            DataNum = ControlCAN.VCI_GetReceiveNum(self.DevType, self.DeviceIndex, self.CANIndex)

        CAN_ReceiveData = (ControlCAN.VCI_CAN_OBJ * num)()
        ReadDataNum = ControlCAN.VCI_Receive(self.DevType, self.DeviceIndex, self.CANIndex, byref(CAN_ReceiveData), num, 0)
        return ReadDataNum, CAN_ReceiveData

    def reset(self):
        nRet = ControlCAN.VCI_ResetCAN(self.DevType, self.DeviceIndex, self.CANIndex)

    def close(self):
        nRet = ControlCAN.VCI_CloseDevice(self.DevType, self.DeviceIndex)


    

# def main():
#     device = GinkGo(0, 0, 2, show=True)
#     device.start()
#     device.read_CAN_status()

#     # device.receive(16)

#     while True:
#         num, data = device.receive(16)
#         for i in range(num):
#             print(data[i].ID, end=' ')
#         print()

#     sleep(5)
#     device.reset()
#     device.close()


# main()