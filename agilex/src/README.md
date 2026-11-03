# How To Start

## 车载端

需要先启动车载端，因为AgilexNav接收map2base_link的变换，中间有odom的桥接，需要等待map2odom初始化完成后才能启动服务器端
  
集成到了一个launch文件中：
``` bash
roslaunch scout_bringup bringup_all.launch
```

## 服务器端

对应本仓库中的server.py
``` bash
conda activate /home/yyw/miniconda3/envs/agilex_nav-py39
```
``` bash
python -m agilex_nav.final_0206.py
```

## win转发端

``` bash
python win_bridge.py
```