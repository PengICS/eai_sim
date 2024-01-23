# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from omni.isaac.cortex.df import DfNetwork, DfState, DfStateMachineDecider, DfStateSequence
from omni.isaac.cortex.dfb import DfBasicContext


class ReachState(DfState):
    def __init__(self, target_p):
        self.target_p = target_p

    def enter(self):
        self.context.robot.arm.send_end_effector(target_position=self.target_p)

    def step(self):
        if np.linalg.norm(self.target_p - self.context.robot.arm.get_fk_p()) < 0.01:
            return None
        return self


def make_decider_network(robot):
    p1 = np.array([0.2, -0.2, 0.01])
    p2 = np.array([0.6, 0.3, 0.6])
    root = DfStateMachineDecider(DfStateSequence([ReachState(p1), ReachState(p2)], loop=True))
    return DfNetwork(root, context=DfBasicContext(robot))
