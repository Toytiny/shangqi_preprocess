import csv
import open3d
import numpy as np
from matplotlib import pyplot as plt
from math import sin, cos, pi
import pandas as pd
import os
from tqdm import tqdm

class Lrr30:

    def __init__(self, path, target_sensor):
        self.data = []
        self.spm_point_cloud = {}

        self.read_raw(path)
        self.draw_spmpcd(target_sensor)
        plt.plot(self.timestamps)
        plt.xlabel('frame')
        plt.ylabel('timestamp')
        plt.savefig('timestamps.png',dpi=500)
        print('Showing Timestamps')
        
    def read_raw(self, path):

        with open(path) as f:
            reader = csv.reader(f)
            self.timestamps = []
            for li, line in enumerate(reader):
                # arrgement version
                if line.pop(0) != 'hasco-lrr30-v1':
                    continue
                line_cp = line[:]
                try:
                    frame = self.dict_frame(line)
                    self.timestamps.append(frame['recvTime'])
                except ValueError:
                    print('{} frame is not compatible'.format(li))
                    if 'frame' not in locals():
                        continue
                    frame['left'] = line

                self.data.append(frame)

    @staticmethod
    def dict_frame(line):
        frame = {}
        # relative time of Aseva receives radar data
        frame['recvTime'] = float(line.pop(0))

        # header
        frame['header'] = header = {}

        header['iEthProtcolVersionMajor'] = int(line.pop(0))
        header['iEthProtcolVersionMirror'] = int(line.pop(0))
        header['iEthTotalPackets'] = int(line.pop(0))
        header['iPlatformSWversion'] = int(line.pop(0))
        header['iPlatformRFversion'] = int(line.pop(0))
        header['iPlatformHWversion'] = int(line.pop(0))
        header['iSensorTotalCount'] = int(line.pop(0))
        header['isMaster'] = int(line.pop(0))
        header['iSensorIndex'] = int(line.pop(0))
        header['iRes'] = int(line.pop(0))
        header['iCurrentFrameTimeStamp'] = int(line.pop(0))
        header['iCurrentFrameID'] = int(line.pop(0))
        line.pop(0)
        header['iSensorIndex_Customer'] = int(line.pop(0))  # TODO

        header['vehicleInfoShow'] = vehicleInfoShow = {}
        vehicleInfoShow['Velocity'] = float(line.pop(0))
        vehicleInfoShow['SteerAngleVal'] = float(line.pop(0))
        vehicleInfoShow['YawRate'] = float(line.pop(0))
        vehicleInfoShow['TurnRadius'] = float(line.pop(0))
        vehicleInfoShow['GearPosVal'] = int(line.pop(0))

        header['splitAA'] = int(line.pop(0))

        header['diagInfoShow'] = diagInfoShow = {}
        diagInfoShow['EcuFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['McuFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['PmicFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['MmicFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['PowerFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['SwFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['CommFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['MemFaulty'] = bool(int(line.pop(0)))

        # spm
        frame['spm'] = spm = {}

        spm['fTimeStamp'] = float(line.pop(0))

        spm['sVdyOutput'] = sVdyOutput = {}
        sVdyOutput['fTimeStamp'] = float(line.pop(0))
        sVdyOutput['fVelocity'] = float(line.pop(0))
        sVdyOutput['fVelocityStd'] = float(line.pop(0))
        sVdyOutput['bVelocityValid'] = bool(int(line.pop(0)))
        sVdyOutput['fYawRate'] = float(line.pop(0))
        sVdyOutput['fYawRateStd'] = float(line.pop(0))
        sVdyOutput['bYawRateValid'] = bool(int(line.pop(0)))
        sVdyOutput['fLngAccel'] = float(line.pop(0))
        sVdyOutput['fLngAccelStd'] = float(line.pop(0))
        sVdyOutput['bLngAccelValid'] = bool(int(line.pop(0)))
        sVdyOutput['fLatAccel'] = float(line.pop(0))
        sVdyOutput['fLatAccelStd'] = float(line.pop(0))
        sVdyOutput['bLatAccelValid'] = bool(int(line.pop(0)))
        sVdyOutput['fSteeringWheelAngle'] = float(line.pop(0))
        sVdyOutput['bSteeringWheelAngleValid'] = bool(int(line.pop(0)))
        sVdyOutput['fCurvature'] = float(line.pop(0))
        sVdyOutput['fCurvatureStd'] = float(line.pop(0))
        sVdyOutput['bCurvatureValid'] = bool(int(line.pop(0)))
        sVdyOutput['eMotionState'] = int(line.pop(0))

        spm['eWaveForm'] = int(line.pop(0))
        spm['iTargetsCount'] = int(line.pop(0))
        spm['iEnvironment_infor'] = int(line.pop(0))

        spm['targets'] = targets = {}
        for i in range(spm['iTargetsCount']):
            targets[i] = target = {}
            target['fAzangle'] = float(line.pop(0))
            target['fElangle'] = float(line.pop(0))
            target['fRange'] = float(line.pop(0))
            target['fPower'] = float(line.pop(0))
            target['fSpeed'] = float(line.pop(0))
            target['fSNR'] = float(line.pop(0))
            target['fRCS'] = float(line.pop(0))
            target['fProbability'] = float(line.pop(0))
            target['uAmbigState'] = int(line.pop(0))
            target['bPeakFlag'] = int(line.pop(0))
            target['fStdRange'] = float(line.pop(0))
            target['fStdSpeed'] = float(line.pop(0))
            target['fStdAzangle'] = float(line.pop(0))
            target['fStdElangle'] = float(line.pop(0))
            target['uStateFlag'] = int(line.pop(0))
            target['iARID'] = int(line.pop(0))
            target['iAVID'] = int(line.pop(0))
            target['iBRID'] = int(line.pop(0))
            target['iBVID'] = int(line.pop(0))

        # spm['SPM_sRadarPara_t'] = SPM_sRadarPara_t = {}
        # SPM_sRadarPara_t['eAntennaType'] = int(line.pop(0))
        # SPM_sRadarPara_t['fStartFreq'] = float(line.pop(0))
        # SPM_sRadarPara_t['fCenterFreq'] = float(line.pop(0))
        # SPM_sRadarPara_t['fBandWidth'] = float(line.pop(0))
        # SPM_sRadarPara_t['fPeriodTime'] = float(line.pop(0))
        # SPM_sRadarPara_t['fRangeAmbig'] = float(line.pop(0))
        # SPM_sRadarPara_t['fVeloAmbig'] = float(line.pop(0))
        # SPM_sRadarPara_t['fAzAngleAmbig'] = float(line.pop(0))
        # SPM_sRadarPara_t['fElAngleAmbig'] = int(line.pop(0))
        # SPM_sRadarPara_t['fRangeRes'] = int(line.pop(0))
        # SPM_sRadarPara_t['fVeloRes'] = float(line.pop(0))
        # SPM_sRadarPara_t['fAzAngleRes'] = float(line.pop(0))
        # SPM_sRadarPara_t['fElAngleRes'] = float(line.pop(0))
        # SPM_sRadarPara_t['uSensorId'] = int(line.pop(0))

        spm['Ptp_Seconds'] = int(line.pop(0))
        spm['Ptp_NanoSeconds'] = int(line.pop(0))
        #
        # # track
        # frame['track'] = track = {}
        #
        # track['fTimeStamp'] = float(line.pop(0))
        # track['iFaultCode'] = int(line.pop(0))
        #
        # track['sObjectList'] = sObjectList = {}
        # sObjectList['fTimeStamp'] = float(line.pop(0))
        # sObjectList['iObjectsNum'] = int(line.pop(0))

        # sObjectList['sObjects'] = sObjects = {}
        # for i in range(sObjectList['iObjectsNum']):

        return frame

    @staticmethod
    def spm2pcd(speed, range, azangle, elangle):
        """
        convert raw data to pcd
        :param speed:
        :param range:
        :param azangle:
        :param elangle:
        :return:
        """
        x = range * cos((azangle) * pi / 180.) * cos((elangle) * pi / 180.)
        y = range * sin((azangle) * pi / 180.) * cos((elangle) * pi / 180.)
        z = range * sin((elangle) * pi / 180.)
        vx = speed * cos((azangle) * pi / 180.)
        vy = speed * sin((azangle) * pi / 180.)

        return x, y, z, vx, vy

    def draw_spmpcd(self, target_sensor):
        """
        draw spm data with .pcd from raw
        :return:
        """

        target_dict = {
            "front": 0,
            "right": 1,
            "left": 4
        }

        target_idx = target_dict[target_sensor]

        for frame in self.data:
            points = []
            sensor_idx = frame['header']['iSensorIndex']

            if sensor_idx == target_idx:
                targets = frame['spm']['targets']
                for i in range(len(targets)):
                    target = targets[i]
                    x, y, z, vx, vy = self.spm2pcd(target['fSpeed'], target['fRange'], target['fAzangle'], target['fElangle'])
                    point = [x, y, z, vx, vy, target['fPower'], target['fRCS'], target['fSpeed']]
                    points.append(point)

                points = np.array(points)
                self.spm_point_cloud[frame['recvTime']] = points

def read_raw_radar(root,save_root,base_ts):

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    lrr = Lrr30(root + '/input/raw/raw.csv', target_sensor="front")

    num_pcs = 0
    num_pnts = 0
    num_void = 0

    for k in lrr.spm_point_cloud:

        current_ts = int(k * 1e3) + base_ts
        v = lrr.spm_point_cloud[k]

        if np.size(v,0)>0:
            num_pnts += len(v[:,0])
            num_pcs +=1
            #dis = np.sqrt(v[:,0]**2+v[:,1]**2+v[:,2]**2)
            #if dis.max()<1000:
            data_path = os.path.join(save_root, str(current_ts).zfill(13) + ".csv")
            data = pd.DataFrame(v)
            data.to_csv(data_path)
            print("saving radar frame: ", str(k))
        else:
            num_void+=1
            # else: 
            #     raise("Not Single Distance Mode")

    avg_pnts = num_pnts/num_pcs
    print('The number of void frames is {}'.format(num_void))
    print('Average points per scan is {}'.format(avg_pnts))
    

def main():

    inhouse_path = "/mnt/12T/fangqiang/"
    save_path = "/mnt/12T/fangqiang/inhouse/"


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    root_path_ls = ["/20220222-10-32-36/"
                 ]
    # utc local
    base_ts_ls = {'20220222-10-32-36': [1645497156380,1645497156628]
                  }
    
    for i in range(len(root_path_ls)):

        root = inhouse_path + root_path_ls[i]
        save_root = save_path + root_path_ls[i] + "/radar_front/"
        ## local base timestamp (meta.xml)
        base_ts = base_ts_ls[root_path_ls[i][1:-1]][1] #[1]
        read_raw_radar(root,save_root,base_ts)
        

if __name__ == '__main__':
    main()