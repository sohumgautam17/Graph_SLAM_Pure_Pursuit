#!/usr/bin/env python3
import rclpy
import numpy as np
import tf2_ros
import math
import csv
import os
from time import gmtime, strftime
from rclpy.node import Node
from numpy import linalg as LA
from os.path import expanduser
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from scipy.spatial import transform, distance
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.issim = False

        drive_topic = '/drive'
        marker_array = '/visualization_marker_array'
        target_marker = '/visualization_marker'

        if not self.issim: # simulation (true)
            odom_topic = 'pf/viz/inferred_pose'
        else: # hardware (false)
            odom_topic = '/ego_racecar/odom'
    
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        # PoseStamped for hardware; Odometry for simulation
        self.odom_sub = self.create_subscription(PoseStamped if not self.issim else Odometry, odom_topic, self.pose_callback, 10)
        self.viz_pub = self.create_publisher(MarkerArray, marker_array, 10)
        self.viz_target_pub = self.create_publisher(Marker, target_marker, 10)

        # Parameters
        self.Kp = 0.35 # proportional gain for steering 
        self.steering_min = -1.571/2  # Min and Max of steering angle
        self.steering_max = 1.571/2  
        
        home = expanduser('~') # path to home dir
        folder_path = '/home/nvidia/sohum_ws/src/pure_pursuit/csv_files' # Change I think
        filename = 'wp-3-14-loop.csv'
        self.csv_file_path = os.path.join(folder_path, filename) # Full path to csv

        waypoints = []
        with open(self.csv_file_path, mode='r') as csvfile: # read mode
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None)  # skip the headers
            for row in csv_reader:
                waypoints.append({
                    'posX': float(row[0]),
                    'posY': float(row[1])
                })

        self.waypoints = waypoints
        self.np_waypoints = np.array([np.array([wp['posX'], wp['posY']]) for wp in self.waypoints]) # Faster computation
        self.L = 2.6  # lookahead distance

        self.visualize_waypoints_static(waypoints)

    # Static Waypoints which are stores in the csv
    def visualize_waypoints_static(self, waypoints):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(waypoints): # waypoint stores x, y 
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.150
            marker.scale.y = 0.15
            marker.scale.z = 0.5
            marker.color.a = 1.0 
            marker.color.r = 1.0  
            marker.color.g = 1.0
            marker.color.b = 1.0  
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = waypoint['posX']
            marker.pose.position.y = waypoint['posY']
            marker.pose.position.z = 0.0
            marker_array.markers.append(marker)

        self.viz_pub.publish(marker_array)
            
        
    def visualize_waypoints(self, waypoints, current_waypoint):
        waypoint = waypoints[current_waypoint]
        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.scale.x = 0.15
        target_marker.scale.y = 0.15
        target_marker.scale.z = 0.5
        target_marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
        target_marker.color.r = 255.0 
        target_marker.color.g = 0.0
        target_marker.color.b = 0.0 
        target_marker.pose.orientation.w = 1.0
        target_marker.pose.position.x = waypoint['posX']
        target_marker.pose.position.y = waypoint['posY']
        target_marker.pose.position.z = 0.0

        self.viz_target_pub.publish(target_marker)


    # Curvature of the arc 
    def pure_pursuit(self, goal_pose):
        '''
        Returns a steering angle given a goal pose in terms of the car's reference frame.

        goal_pose: Tuple (x, y) of the goal pose
        return: Float of desired steering angle
        '''

        L2 = goal_pose[0] ** 2 + goal_pose[1] ** 2
        gamma = 2 * abs(goal_pose[1]) / L2
        angle = self.Kp * gamma

        # why do we multiple by the y coordinate
        angle *= np.sign(goal_pose[1])

        angle = max(self.steering_min, angle)
        angle = min(self.steering_max, angle)

        return angle
        
    # What is the point of this function
    def transform_goal_point(self, x_goal, y_goal, x_current, y_current, theta):
        x_diff = x_goal - x_current
        y_diff = y_goal - y_current
        x_goal_car = x_diff * math.cos(theta) + y_diff * math.sin(theta)
        y_goal_car = -x_diff * math.sin(theta) + y_diff * math.cos(theta)

        return x_goal_car, y_goal_car
    
    def pose_callback(self, pose_msg):
        if self.issim:
            posX = pose_msg.pose.pose.position.x
            posY = pose_msg.pose.pose.position.y
            quat = pose_msg.pose.pose.orientation
        else:
            posX = pose_msg.pose.position.x
            posY = pose_msg.pose.position.y
            quat = pose_msg.pose.orientation

        quat = [quat.x, quat.y, quat.z, quat.w]
        euler = euler_from_quaternion(quat)
        theta = euler[2]

        lookahead_pos = (posX + self.L * math.cos(theta), posY + self.L * math.sin(theta))

        np_current_pos = np.zeros_like(self.np_waypoints)
        np_current_pos[:, 0] = lookahead_pos[0]
        np_current_pos[:, 1] = lookahead_pos[1]
        dist = np.linalg.norm(self.np_waypoints - np_current_pos, axis=1)
        closest_wp_index = np.argmin(dist)
        closest_wp = self.waypoints[closest_wp_index]

        self.visualize_waypoints(self.waypoints, closest_wp_index)

        x_diff = closest_wp['posX'] - posX
        y_diff = closest_wp['posY'] - posY       

        x_goal_car = x_diff * math.cos(theta) + y_diff * math.sin(theta)
        y_goal_car = -x_diff * math.sin(theta) + y_diff * math.cos(theta)
        
        steering_angle = self.pure_pursuit((x_goal_car, y_goal_car))

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 1.5
        drive_msg.drive.steering_angle = steering_angle

        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()