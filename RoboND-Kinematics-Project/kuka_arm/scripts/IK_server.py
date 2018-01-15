#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
import numpy as np
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *

class CalcObject:
   # contants for triangle
   side_a = 1.501
   side_c = 1.25
   side_a2 = side_a * side_a
   side_c2 = side_c * side_c
   side_2ac = side_a * side_c
   half_pi = np.pi / 2

   def __init__(self):
        #
        # Correction Needed to Account of Orientation Diff Between Def of Gripper Link in URDF vs DH
        #
        R_z = Matrix([[ cos(np.pi),     -sin(np.pi),    0,      0],
                      [ sin(np.pi),      cos(np.pi),    0,      0],
                      [          0,               0,    1,      0],
                      [          0,               0,    0,      1]])

        R_y = Matrix([[ cos(-np.pi/2),            0, sin(-np.pi/2),     0],
                      [             0,            1,             0,     0],
                      [-sin(-np.pi/2),            0, cos(-np.pi/2),     0],
                      [          0,               0,             0,     1]])

        # Create symbols
        q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
        d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
        a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')
        #
        # Create Modified DH parameters
        #
        s = { alpha0:    0,     a0:     0, d1:  0.75, q1:q1,
              alpha1:-pi/2,     a1:  0.35, d2:     0, q2: q2-pi/2,
              alpha2:    0,     a2:  1.25, d3:     0, q3:q3,
              alpha3:-pi/2,     a3:0.0536, d4:  1.50, q4:q4,
              alpha4: pi/2,     a4:     0, d5:     0, q5:q5,
              alpha5:-pi/2,     a5:     0, d6:     0, q6:q6,
              alpha6:    0,     a6:     0, d7: 0.303, q7:  0}

       #
        # Define Modified DH Transformation matrix
        #
        T0_1 = self.get_DH_matrix(q1, alpha0, a0, d1).subs(s)
        T1_2 = self.get_DH_matrix(q2, alpha1, a1, d2).subs(s)
        T2_3 = self.get_DH_matrix(q3, alpha2, a2, d3).subs(s)
        T3_4 = self.get_DH_matrix(q4, alpha3, a3, d4).subs(s)
        T4_5 = self.get_DH_matrix(q5, alpha4, a4, d5).subs(s)
        T5_6 = self.get_DH_matrix(q6, alpha5, a5, d6).subs(s)
        T6_G = self.get_DH_matrix(q7, alpha6, a6, d7).subs(s)
        #
        # Create individual transformation matrices
        #
        T0_G = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_G
        R_corr = simplify(R_z * R_y)
        self.mT_total = simplify(T0_G * R_corr)
        self.mR0_3_Mat = T0_1[0:3,0:3] * T1_2[0:3,0:3] * T2_3[0:3,0:3]
        #
        # Extract rotation matrices from the transformation matrices
        #
        # Roll, Pitch, Yaw
        r, p, y = symbols('r p y')
        ROT_x = Matrix([[1,     0,               0],
                        [0,     cos(r),    -sin(r)],
                        [0,     sin(r),     cos(r)]]) #Roll
        ROT_y = Matrix([[cos(p),    0,      sin(p)],
                        [0,         1,           0],
                        [-sin(p),   0,      cos(p)]]) #Pitch
        ROT_z = Matrix([[cos(y),  -sin(y),  0],
                        [sin(y),   cos(y),      0],
                        [     0,        0,      1]]) #Yaw
        ROT_EE = ROT_z * ROT_y * ROT_x
        Rot_Error = ROT_z.subs(y, np.pi) * ROT_y.subs(p, -np.pi/2)
        self.mROT_EE = ROT_EE * Rot_Error

   def get_DH_matrix(self, q, alpha, a, d):
        return Matrix([[            cos(q),           -sin(q),           0,     a],
                       [ sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d ],
                       [ sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d ],
                       [                 0,                 0,           0,              1]])

   def get_angles(self, roll, pitch, yaw, EE):
        #
        # Calculate joint angles using Geometric IK method
        #
        ROT_EE = self.mROT_EE.subs({'r': roll, 'p':pitch, 'y':yaw})
        WC = EE - (0.303) * ROT_EE[:,2]

        # Calculate joint angles
        theta1 = atan2(WC[1], WC[0])

        # triangle for theta2 and theta3
        radius  = sqrt(WC[0]*WC[0]+WC[1]*WC[1]) - 0.35 # a1: 0.35
        side_b2 = radius**2 + (WC[2]-0.75)**2 # d1: 0.75
        side_b  = sqrt(side_b2)
        angle_a = acos((side_b2 + CalcObject.side_c2 - CalcObject.side_a2) / (2 * side_b * CalcObject.side_c))
        angle_b = acos((CalcObject.side_a2 + CalcObject.side_c2 - side_b2) / CalcObject.side_2ac)
        #angle_c = acos((CalcObject.side_a2 + side_b2 - CalcObject.side_c2) / (2 * CalcObject.side_a * side_b))

        theta2 = CalcObject.half_pi - angle_a - atan2(WC[2] - 0.75, radius)
        theta3 = CalcObject.half_pi - (angle_b + 0.036)  # 0.036 accounts for sag in link4 of -0.054m

        R0_3 = self.mR0_3_Mat.evalf(subs={'q1': theta1, 'q2': theta2, 'q3': theta3})

        R3_6 = R0_3.T * ROT_EE # use transpose matrix instead of inverse matrix to save computation

        #Euler angles from rotation matrix
        theta4 = atan2(R3_6[2,2], -R3_6[0,2])
        theta5 = atan2(sqrt(R3_6[0,2]**2 + R3_6[2,2]**2), R3_6[1,2])
        theta6 = atan2(-R3_6[1,1], R3_6[1,0])
        return theta1, theta2, theta3, theta4, theta5, theta6, WC

   def get_T_total(self):
        return self.mT_total;


# create a calc object
_calcObj = CalcObject()



def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

	    # Extract end-effector position and orientation from request
	    # px,py,pz = end-effector position
	    # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            ### Your IK code here
	    # The calculaion has been done in the Calc Object as we want to reuse calculation as much as possible
            ###
            EE = Matrix([[px], [py], [pz]])
            theta1, theta2, theta3, theta4, theta5, theta6, your_wc = _calcObj.get_angles(roll, pitch, yaw, EE)

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
	    joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
	    joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
