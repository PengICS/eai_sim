# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from omni.isaac.cortex.df import DfNetwork, DfDecider, DfAction, DfDecision
from omni.isaac.cortex.dfb import DfRobotApiContext


class Context(DfRobotApiContext):
    def __init__(self, robot):
        super().__init__(robot)

        self.reset()
        self.add_monitors([Context.monitor_y, Context.monitor_is_left, Context.monitor_is_middle])

    def reset(self):
        self.y = None
        self.is_left = None
        self.is_middle = None

    def monitor_y(self):
        self.y = self.robot.arm.get_fk_p()[1]

    def monitor_is_left(self):
        self.is_left = self.y < 0

    def monitor_is_middle(self):
        self.is_middle = -0.15 < self.y and self.y < 0.15


class PrintAction(DfAction):
    def __init__(self, msg=None):
        super().__init__()
        self.msg = msg

    def enter(self):
        if self.params is not None:
            print(self.params)
        else:
            print(self.msg)


class Dispatch(DfDecider):
    def __init__(self):
        super().__init__()
        self.add_child("print_left", PrintAction("<left>"))
        self.add_child("print_right", PrintAction("<right>"))
        self.add_child("print", PrintAction())

    def decide(self):
        if self.context.is_middle:
            return DfDecision("print", "<middle>")  # Send parameters down to generic print.

        if self.context.is_left:
            return DfDecision("print_left")
        else:
            return DfDecision("print_right")


def make_decider_network(robot):
    return DfNetwork(Dispatch(), context=Context(robot))
