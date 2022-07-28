# 数据格式说明 v0.1.3

## 目录 (* 代表无用)
├── input \
│   ├── audio * `音频数据，目前为空`\
│   ├── bus * `LiDar原始数据，后面不会上传`\
│   ├── lidar `LiDar .pcd 数据，以相对时间戳命名`\
│   ├── raw `其他传感器原始数据，radar数据就存在这里，需要通过 read_raw.py 来处理` \
│   └── video `image数据，原本是.avi视频格式，会切分成 .jpg 或 .png` \
├── meta.xml `储存采集软件信息，采集时间0点在这里提取` \
├── output \
│   └── online `\sample 下的 gnssimu-sample-v6@1 存放惯导数据 ` \
└── project.aspro *

## 新增
1. 正前 camera 内外参
2. 修改了LiDar的外参，跟6.25日发的 “1号车内外参.txt” 统一

### 1. 采集时间零点

目前radar与LiDar数据的时间戳都是相对时间戳，这里相对的时间0点是 `meta.xml` 
文件第一行的 `start_posix_local` 储存的13位unix时间戳

### 2. 从 raw.csv 中提取 radar 数据

read_raw.py 遵照 hasco LRR30 系列毫米波雷达的数据协议提取并保存 radar 数据， 
包含 radar 状态，spm 点云数据，object 雷达目标物数据以及一些必要的debug信息。
几乎所有从雷达输出的数据你都可以在 Lrr30 类中得到。

```python
from read_raw import Lrr30

lrr = Lrr30('raw.csv')
lrr.data  # radar 全部数据，dict
lrr.spm_point_cloud  # radar spm 点云数据 np.array by [x, y, z, vx, vy, power, rcs]
# TODO
# 解决object 雷达目标物数据提取不正确的问题
# 点云数据保存成 .pcd 格式
# 点云坐标系转换
# 点云拼接
```

### 3. 传感器内外参

以前保中心在地面的投影为 (0, 0, 0) 点

| | x (m) | y (m) | z (m) | yaw (°) | pitch (°) | roll (°) |
| :-----| ----: | :----: | :----: | :----: | :----: | :----: |
| 128 LiDar | -2.50 | 0 | 2.03 | 4.9 | -1.5 | 0 |
| 正前 radar | 0.06 | -0.2 | 0.7 | 0 | 3.19 | 180 |
| 左前 radar | -0.34 | 0.94 | 0.7 | 61 | 0 | 0 |
| 右前 radar | -0.34 | -0.94 | 0.7 | -61.5 | 0 | 180 |
| 正前 camera | -1.793 | -0.036 | 1.520 | -91.66 | -0.09 | -90.9 | 

正前相机内参

cam_intrin_B_1 = dict({"fx": 1146.50085849981, "fy": 1146.58873864681, "cx": 971.982356795787,
                       "cy": 647.093113465516,  "k1": 3.37137465120156, "k2": 10.0609165093087,
                       "p1": 0.00113759316537766, "p2": -0.000160764084280244, "k3": 1.44362350707298,
                       "k4": 3.75609517809686, "k5": 11.3033094103828, "k6": 4.64261770650288})

### 4. 雷达性能参数

我们使用的雷达分两种，LRR30-F远距离成像雷达（正前） 和 LRR30-C中距离成像雷达（左前、右前）

1. LRR30-F远距离成像雷达

    初始工作在近距离模式下，根据载体速度进行切换，近距离模式下，当载体速度大于115km/h，切换至远距离模式。远距离模式下，当载体速度低于110km/h切换至近距离模式。

   （因特殊需要，目前正前为mid-mid模式）
```yaml
工作频段 Frequeny Band: 76 ~ 77 GHz
带宽 BW:  <1000 MHz
距离 Range:  
  最大探测 Maximum: 285 m
  最小探测 Minimum: 0.25 m
  距离分辨率 Range Resolution: 0.2m (<75m), 0.4m (>75m & <150m), 0.8m (>150m)
  距离分离度 Range Separation: 优于0.4m (<75m), 优于0.8m (>75m & <150m), 优于1.6m (>150m)
  距离精度 Range Accuracy: ±0.1m (<75m), ±0.2 (>75m & <150m), ±0.4m (>150m)
相对径向速度 Speed Radial:
  速度范围 Radial Speed Range: -85 ~ 45 m/s
  速度分辨率 Speed Resolution: 0.13 m/s 
  速度分离度 Speed Separation: 0.26 m/s
  速度精度 Speed Accuracy: ±0.06 m/s
角度 Angle:
  水平角度范围 Az Angle FOV: ±60deg(40m), ±12deg(230m)
  俯仰角度范围 Ei Angle FOV: ±10deg(200m)
  角度分离度 Angle Separation: 2deg
  角度精度 Angle Accuracy: 
    Az: ±0.2deg @±12deg, ±0.4deg @±45deg, ±0.8deg @±60deg
    Ei: ±0.5deg @±10deg
```

2. RR30-C中距离成像雷达

    只有一种工作模式

```yaml
工作频段 Frequeny Band: 76 ~ 77 GHz
带宽 BW:  <1000 MHz
距离 Range:  
  最大探测 Maximum: 125 m
  最小探测 Minimum: 0.2 m
  距离分辨率 Range Resolution: 0.1765m (<65m), 0.3333m (>65m)
  距离分离度 Range Separation: 优于0.35m (<75m), 优于0.67m (>65m)
  距离精度 Range Accuracy: ±0.1m (<65m), ±0.2m (>65m)
相对径向速度 Speed Radial:
  速度范围 Radial Speed Range: -65 ~ 65 m/s
  速度分辨率 Speed Resolution: 0.13 m/s 
  速度分离度 Speed Separation: 0.26 m/s
  速度精度 Speed Accuracy: ±0.06 m/s
角度 Angle:
  水平角度范围 Az Angle FOV: ±60deg(90m), ±45deg(125m)
  俯仰角度范围 Ei Angle FOV: ±10deg(100m)
  角度分离度 Angle Separation: 
    Az: 2deg
    Ei: 3deg
  角度精度 Angle Accuracy: 
    Az: ±0.4deg @±45deg, ±0.8deg @±60deg
    Ei: ±0.3deg @±10deg, ±0.5deg @±20deg
```


 


