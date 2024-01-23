import asyncio
import omni
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.
from omni.isaac.core import World

from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka.controllers import PickPlaceController
import numpy as np
import socket

# omni.usd.get_context().open_stage("/home/liugc/Desktop/navigation/house.usd")

simulation_app.update()

print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading

while is_stage_loading():
    simulation_app.update()
print("Loading Complete")

world = World()
# We add the task to the world here
world.scene.add_default_ground_plane()
# goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])

goal_position = {
    "red_cube":np.array([-0.3, -0.3, 0.0515 / 2.0]),
    "yellow_cube":np.array([-0.3, -0.4, 0.0515 / 2.0]),
    "green_cube":np.array([-0.3, -0.5, 0.0515 / 2.0]),
}
RedCube = world.scene.add(DynamicCuboid(prim_path="/World/red_cube",
                                            name="red_cube",
                                            position=np.array([0.3, 0.3, 0.3]),
                                            scale=np.array([0.0515, 0.0515, 0.0515]),
                                            color=np.array([0.7, 0.0, 0.0])))
YellowCube = world.scene.add(DynamicCuboid(prim_path="/World/yellow_cube",
                                            name="yellow_cube",
                                            position=np.array([0.3, 0.4, 0.3]),
                                            scale=np.array([0.0515, 0.0515, 0.0515]),
                                            color=np.array([0.7, 0.7,  0.0])))
GreenCube = world.scene.add(DynamicCuboid(prim_path="/World/green_cube",
                                            name="green_cube",
                                            position=np.array([0.3, 0.5, 0.3]),
                                            scale=np.array([0.0515, 0.0515, 0.0515]),
                                            color=np.array([ 0.0, 0.7,  0.0])))
cube = RedCube
cube_name = "red_cube"
franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka",
                                       name="fancy_franka"))
world.reset()



franka = world.scene.get_object("fancy_franka")


HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65530  # Port to listen on (non-privileged ports are > 1023)
for i in range(500):
    world.step(render=True)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while  True:
            data = conn.recv(1024)
            if not data:
                break
            cube_name = data.decode('utf-8')
            
            if cube_name == "red_cube":
                cube = RedCube
            elif cube_name == "green_cube":
                cube = GreenCube
            else:
                cube = YellowCube
            conn.sendall(data)

            # If simulation is stopped, then exit.
            controller = PickPlaceController(
                name="pick_place_controller",
                gripper=franka.gripper,
                robot_articulation=franka,)
            
            franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

            while simulation_app.is_running():
                cube_position, _ = cube.get_world_pose()
                current_joint_positions = franka.get_joint_positions()
                observations = {
                    franka.name: {
                        "joint_positions": current_joint_positions,
                    },
                    cube.name: {
                        "position": cube_position,
                        "goal_position": goal_position[cube.name]
                    }
                }
                current_observations = observations
                actions = controller.forward(
                    picking_position=current_observations[cube_name]["position"],
                    placing_position=current_observations[cube_name]["goal_position"],
                    current_joint_positions=current_observations["fancy_franka"]["joint_positions"],
                )
                franka.apply_action(actions)
                # franka.gripper.close()
                
                world.step(render=True)
                if controller.is_done() :
                    # world.reset()
                    franka.post_reset()
                    controller.resume()
                    # world.pause()
                    print("finished pick and place")
                    
                    break
                    
            if not world.is_playing:
                world.step(render=True)
                continue
            
        simulation_app.close()