# LAB2 README 本机运行步骤

工作区：`/home/swx/COMP0246_S26_Labs`。使用 ROS2 Humble 与系统 Python 3.10。
已有代码无需再次 clone 或 git pull。

## Step 0：安装依赖、构建和启动

在本机终端执行，sudo 提示时输入本机密码。以下为原 README 的依赖清单，另外补充代码绘图需要的 matplotlib。

```bash
sudo apt-get update
sudo apt-get install -y \
  ros-humble-python-orocos-kdl-vendor \
  ros-humble-orocos-kdl-vendor \
  ros-humble-kdl-parser \
  ros-humble-joint-state-publisher-gui \
  ros-humble-controller-manager \
  ros-humble-joint-trajectory-controller \
  ros-humble-joint-state-broadcaster \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  python3-matplotlib

cd /home/swx/COMP0246_S26_Labs
export PATH="/usr/bin:/bin:$PATH"
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

每次打开新终端，先执行：

```bash
cd /home/swx/COMP0246_S26_Labs
export PATH="/usr/bin:/bin:$PATH"
source /opt/ros/humble/setup.bash
source install/setup.bash
```

终端一启动仿真，并保持运行：

```bash
ros2 launch trajectory_demo bringup.launch.py \
  urdf_package:=franka_description \
  urdf_package_path:=urdfs/fr3.urdf \
  rviz_config_package:=trajectory_demo \
  rviz_package_path:=cfg/rviz.rviz
```

终端二初始化环境后，检查两个控制器均为 active：

```bash
ros2 control list_controllers
```

按 README 手动发布一条关节轨迹，检查机械臂是否移动：

```bash
ros2 topic pub -1 /franka_arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7'], points: [{positions: [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], time_from_start: {sec: 1}}]}"
```

## Step 1：执行默认笛卡尔目标并绘图

终端一保持运行，在终端二执行：

```bash
ros2 run trajectory_demo traj_demo --ros-args \
  -p base_link:=base \
  -p ee_link:=fr3_link8 \
  -p traj_cfg_pkg:=trajectory_demo \
  -p traj_cfg_path_within_pkg:=cfg/traj_waypoints.yaml
```

到达目标并出现 `Trajectory complete` 后，在终端二按一次 Ctrl+C，保存并显示图像。
当前代码将图像保存在 `/home/swx/Desktop`。

提交：执行轨迹图。

## Step 2：运行各组员姓名首字母并计时

当前已有 S、D、W 配置。每次只运行一个节点，完成并 Ctrl+C 后再运行下一个。

```bash
ros2 run trajectory_demo traj_demo --ros-args \
  -p base_link:=base -p ee_link:=fr3_link8 \
  -p traj_cfg_pkg:=trajectory_demo \
  -p traj_cfg_path_within_pkg:=cfg/S.yaml
```

将 `cfg/S.yaml` 分别改为 `cfg/D.yaml`、`cfg/W.yaml` 即可运行其他字母。
S 的计划总时长为 17 秒，D、W 为 18 秒；实际时长读取 `Measured execution time` 日志。

提交：每名组员的字母轨迹图、对应 YAML 配置和实际运行时间。

## Step 3：加入姿态目标

```bash
ros2 run trajectory_demo traj_demo --ros-args \
  -p base_link:=base -p ee_link:=fr3_link8 \
  -p traj_cfg_pkg:=trajectory_demo \
  -p traj_cfg_path_within_pkg:=cfg/S_orientation.yaml
```

该配置复用 S 的位置与时间，姿态目标为 RPY `[0.0, 0.2, 0.0]` 弧度。
完成后 Ctrl+C，将另外生成位置和姿态随时间变化的图。
README 提及的 `traj_demo_student.py` 在本工作区不存在；已有姿态实现位于 `traj_demo.py`，仍使用 `traj_demo` 入口运行。

提交：位置图、姿态图与配置。

## Step 4：报告讨论

写约半页文字，说明有障碍物且工具姿态影响切割质量时，可以采用哪些控制策略，以及它们与本实验方法的区别。README 明确要求描述，无需实现。

## 修改后的重建

修改源码或 YAML 后，在工作区根目录执行：

```bash
colcon build --packages-select trajectory_demo
source install/setup.bash
```

`setup.py` 已适配为安装 cfg 目录中的全部 YAML，因此可直接使用上面的 `cfg/S.yaml` 等相对包路径。
