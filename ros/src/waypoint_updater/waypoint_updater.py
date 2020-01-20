#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.kdtree import KDTree
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
STOP_LINE_MARGIN = 2
CONSTANT_DECEL = 1 / LOOKAHEAD_WPS

MAX_DECEL = 0.5
LOGGING_THROTTLE_FACTOR = 1

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)        

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.pose = None
        self.base_waypoint = None
        self.stopline_wp_idx = -1
        self.waypoints_tree = None
        self.spin_rate = 10
        self.decelerate_count = 0        

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.spin()

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_next_waypoint()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoint.waypoints[closest_idx:farthest_idx]

        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        stop_idx = max(self.stopline_wp_idx - closest_idx - STOP_LINE_MARGIN, 0)
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            # Distance includes a number of waypoints back so front of car stops at line
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        self.decelerate_count += 1
        if (self.decelerate_count % LOGGING_THROTTLE_FACTOR) == 0:
            size = len(waypoints) - 1
            vel_start = temp[0].twist.twist.linear.x
            vel_end = temp[size].twist.twist.linear.x
            rospy.logwarn("DECEL: vel[0]={:.2f}, vel[{}]={:.2f}, stopline_wp_idx={}, closest_idx={}".format(vel_start, size, vel_end, self.stopline_wp_idx, closest_idx))
        return temp        

    def spin(self):
        rate = rospy.Rate(self.spin_rate)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoint:
                self.publish_waypoints()
            rate.sleep()

    def get_next_waypoint(self):
        pose_x = self.pose.pose.position.x
        pose_y = self.pose.pose.position.y

        closest_ids = self.waypoints_tree.query([pose_x, pose_y], 1)[1]

        closest_coords = self.waypoints_tree.data[closest_ids]
        prev_coords = self.waypoints_tree.data[closest_ids - 1]

        cl_vect = np.array(closest_coords)
        prev_vect = np.array(prev_coords)
        pose_vect = np.array([pose_x, pose_y])

        val = np.dot(cl_vect - prev_vect, pose_vect - cl_vect)
        if val > 0:
            closest_ids = (closest_ids + 1) % len(self.waypoints_tree.data)
        return closest_ids



    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoint = waypoints
        if not self.waypoints_tree:
            waypoint_coord = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(waypoint_coord)

    def traffic_cb(self, msg):
        if self.stopline_wp_idx != msg.data:
            rospy.logwarn(
                "LIGHT: new stopline_wp_idx={}, old stopline_wp_idx={}".format(msg.data, self.stopline_wp_idx))
            self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
