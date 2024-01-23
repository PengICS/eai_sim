import asyncio
from mavsdk import System


class IsaacSimWrapper:

    def __init__(self, server_port=50040):
        self.drone = System(port=server_port)
            # mavlink

    async def connect(self, port=14550):
        await self.drone.connect(system_address="udp://:" + str(port))
        # await self.drone.connect(system_address="tcp://:" + str(4562))
        print("waiting for drone connect...")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("drone connected")
                break

        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- global posttion is good enough to fly...")
                break
        async for teerrain_info in self.drone.telemetry.home():
            self.absolute_altitude = teerrain_info.absolute_altitude_m
            self.absolute_latitude = teerrain_info.latitude_deg 
            self.absolute_longitude = teerrain_info.longitude_deg
            break

    def arm(self):
        asyncio.get_event_loop().run_until_complete(self.drone.action.arm())

    def takeoff(self):
        asyncio.get_event_loop().run_until_complete(self.drone.action.takeoff())
    
    def land(self):
        asyncio.get_event_loop().run_until_complete(self.drone.action.land())

    def back(self):
        asyncio.get_event_loop().run_until_complete(self.drone.action.return_to_launch())

    def fly_to(self,position):
        asyncio.get_event_loop().run_until_complete(self.drone.action.goto_location(position[0] , position[1] , position[2], 0))

    def fly_path(self,points) :
        return
    
    def get_position(self):
        position = asyncio.get_event_loop().run_until_complete(self.drone.telemetry.position().__aiter__().__anext__())
        return [position.latitude_deg, position.longitude_deg,position.absolute_altitude_m]
    
    def get_drone_position(self):
        position = asyncio.get_event_loop().run_until_complete(self.drone.telemetry.position().__aiter__().__anext__())
        return [position.latitude_deg, position.longitude_deg,position.absolute_altitude_m]
    
    def set_yaw(self, yaw):
        self.drone.action.set_actuator(yaw)

    def get_yaw(self):
        self.drone.telemetry.actuator_output_status()