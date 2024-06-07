import argparse
import omni

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to ISAAC SIM: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument(
    "--usd_path", type=str, help="Path to usd file, should be relative to your default assets folder", default="/home/liugc/python_workspace/pcl_sim/px4_sim/env/house.usd",required=False
)
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


# Locate Isaac Sim assets folder to load sample
from omni.isaac.core.utils.nucleus import is_file

from omni.isaac.core.world import World
import omni.isaac.core.utils.prims as prim_utils
from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.mavlink_backend import MavlinkBackend, MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

import numpy as np


# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation


"""
Main
"""

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()

        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Add a custom light with a high-definition HDR surround environment of an exhibition hall,
        # instead of the typical ground plane
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Simple Room"])

        # prim_utils.create_prim(
        #     "/World/Light/DomeLight",
        #     "DomeLight",
        #     attributes={
        #         "texture:file": "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr",
        #         "intensity": 1000.0
        # })

        #self.pg.set_viewport_camera([1.0, 5.15, 1.65], [0.0, -1.65, 3.3])


        for i in range(2):
            self.vehicle_factory(i, gap_x_axis=4.5)
        self.pg._world.reset()
        self.world.reset()
        self.stop_sim = False

    def vehicle_factory(self, vehicle_id: int, gap_x_axis: float):
        """Auxiliar method to create multiple multirotor vehicles

        Args:
            vehicle_id (_type_): _description_
        """
        config_multirotor = MultirotorConfig()
        
        mavlink_config = MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            #"connection_ip": "192.168.63.138",
            #"connection_baseport": 50051,
            "px4_autolaunch": True,
            "px4_dir": "/home/liugc/source_code/PX4-Autopilot",
            "px4_vehicle_model": 'iris'
        })
        config_multirotor.backends = [MavlinkBackend(mavlink_config)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            vehicle_id,
            [-5.0, 0.00 + gap_x_axis*vehicle_id, 1],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

    def run(self):
        
        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        
        # Cleanup and stop
        self.timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    
    pg_app = PegasusApp()
    pg_app.run()
