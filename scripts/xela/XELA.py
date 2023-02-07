import sys
import os
sys.path.append(os.path.dirname(__file__))

from subprocess import call
from textwrap import indent

from matplotlib.pyplot import get
from GinkGo import GinkGo
import numpy as np

class XELA():
    def __init__(self):
        self.device = GinkGo(0, 0, 2, show=True)
        self.init_position = np.stack(np.meshgrid(np.arange(4), np.arange(4), indexing='ij'), axis=-1) * 0.0047
        self.init_position = np.concatenate([self.init_position, np.zeros_like(self.init_position[:, :, [0]])], axis=-1)

    def start(self):
        self.device.start()
        self.device.read_CAN_status()
        self.init_frame_buffer = self.read()

    def read(self):
        num, data_arr = self.device.receive(16)
        frame_buffer = np.zeros((4, 4, 3), dtype=np.uint16)
        for i in range(num):
            #sdanumber(12:8), taxelnumber(8:4), mid(4:0)
            id = data_arr[i].ID
            sda_id = (id >> 8) & 0xf
            taxel_id = (id >> 4) & 0xf
            micro_id = id & 0xf
            data = data_arr[i].Data
            data_len = data_arr[i].DataLen

            row = taxel_id // 4
            col = taxel_id % 4
            row += sda_id * 2
            frame_buffer[row, col, 0] = data[1] & 0xff
            frame_buffer[row, col, 0] = (frame_buffer[row, col, 0] << 8) + (data[2] & 0xff)
            frame_buffer[row, col, 1] = data[3] & 0xff
            frame_buffer[row, col, 1] = (frame_buffer[row, col, 1] << 8) + (data[4] & 0xff)
            frame_buffer[row, col, 2] = data[5] & 0xff
            frame_buffer[row, col, 2] = (frame_buffer[row, col, 2] << 8) + (data[6] & 0xff)
        return frame_buffer / 1000000

    def get(self):
        offset = self.read() - self.init_frame_buffer
        return self.init_position + offset[:, :, [1, 0, 2]]


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    xela = XELA()
    xela.start()
    print("debug0")
    while True:
        frame = xela.get()
        print('Debug')
        # print(frame[:, :, 2])
    
        plt.ion()
        plt.clf()
        ax = plt.subplot(111)
        s = (frame[:, :, 2] + 0.0005) * 1000000
        ax.scatter(frame[:, :, 0].reshape(-1), frame[:, :, 1].reshape(-1), s=s)
        ax.set_xlim(-0.0047, 4*0.0047)
        ax.set_ylim(-0.0047, 4*0.0047)
        plt.waitforbuttonpress(0.01)

if __name__ == '__main__':
    main()