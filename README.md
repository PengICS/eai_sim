# eai_sim
一、仿真软件Isaac sim安装
Isaac sim的安装和使用文档参考 Isaac sim 教程 
安装完成后设置环境变量：
# Isaac Sim root directory
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
在VS CODE工程目录下新建软链
# enter the cloned repository
cd my_workspace
# create a symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim

二、PX4无人机插件安装
Isaac sim的初始环境不支持PX4无人机，需要在Isaac sim安装一个插件。安装的官方文档参考https://pegasussimulator.github.io/PegasusSimulator/source/setup/installation.html
1. 在isaac sim上安装插件
[图片]

2. 在ubuntu本地编译PX-4无人机代码
[图片]
3. 最后把本地编译好的PX4路径配置到Isaac sim环境。
三、PX-4无人机仿真环境的启动
在VS code下运行项目中house.py文件，启动过程，terminal有以下界面，这里面包含了每台无人机的控制端口，无人机的控制通过mavlink协议。
[图片]
chatgpt和无人机
启动chatgpt进程只需要运行 chatgpt.py文件，启动连接成功后只需要在terminal和chagpt交互控制。
[图片]
由于代码里命名了两台无人机为aw1和aw2，所以和chatgpt对话的时候需要告诉chatgpt两台无人机的名字。
交互文本示例： 
we have two drone named aw1和aw2, aw1 and aw2 arm
可以让两台无人机准备好飞行
aw1 and aw2 take off
可以让两台无人机起飞
aw1 fly left , aw2 fly up 
可以让一台无人机左边飞，一台往上飞。
aw1 and aw2 go back
可以让无人机返回原点。
无人机的接口：
[图片]
chatgpt代理的地址
由于现有的openai key只能在代理中使用，需要配置一下chatgpt代理的地址
[图片]
四、机械臂与chatgpt仿真环境
启动仿真环境
在vscode环境下，运行PickAndPlaceExample.py文件，就可以启动仿真环境。
启动chatgpt
在vscode下运行chatgpt_manulaptor.py文件就可以启动连接上仿真环境和chatgpt，在terminal和chatgpt交互。
交互文本示例
Franka pick the green cube
Franka pick the yellow cube
