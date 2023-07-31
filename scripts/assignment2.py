#!/usr/bin/env python

import rospy
import numpy as np
import argparse

from geometry_msgs.msg import PoseWithCovarianceStamped
import dynamic_reconfigure.client

import matplotlib.pyplot as plt
from nav_msgs.srv import GetMap

import cv2
import tf
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

#imports for costmap
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate

import time

class TurtleBot:
    def __init__(self):
        self.initial_position = None
        self.number_of_updates = 0
        self.shape = None
        self.cost_map = None
        self.no_walls = None

        self.replan_distance = 9
        # self.tmp_map


        self.position = None

        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, callback=self.set_initial_position)
        rospy.wait_for_service('static_map')
        static_map = rospy.ServiceProxy('static_map', GetMap)
        self.map_data = static_map().map
        self.map_org = np.array([self.map_data.info.origin.position.x, self.map_data.info.origin.position.y])
        shape = self.map_data.info.height, self.map_data.info.width
        self.map_arr = np.array(self.map_data.data, dtype='float32').reshape(shape)
        self.resolution = self.map_data.info.resolution

        # rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, self.init_costmap_callback)

        self.cost_map = np.zeros_like(self.map_arr)

        rospy.Subscriber('/move_base/global_costmap/costmap_updates', OccupancyGridUpdate, self.costmap_callback_update)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.get_position_callback)
        self.client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        # Waits until the action server has started up and started listening for goals.

        # time.sleep(0.5)
        # tmp_map = np.copy(self.map_arr)
        # tmp_map = self.dilate_walls(tmp_map, 5)
        # self.no_walls = self.cost_map - tmp_map

        self.client.wait_for_server()


        print("Waiting for an initial position...")
        while self.initial_position is None:
            continue
        print("The initial position is {}".format(self.initial_position))


    def get_position_callback(self, msg):
        pose = msg.pose.pose
        self.position = np.array([pose.position.x, pose.position.y])

    def position_to_map(self, pos):
        return (pos - self.map_org) // self.resolution

    def set_initial_position(self, msg):
        initial_pose = msg.pose.pose
        self.initial_position = np.array([initial_pose.position.x, initial_pose.position.y])

    def show_map(self, point=None):
        if not self.cost_map is None:
            map1 = self.map_arr
            map2 = self.cost_map

            fig, ax = plt.subplots(1, 3, figsize=(15,5))
            ax[0].imshow(map1)
            ax[0].set_title('Map Array')
            ax[1].imshow(map2)
            ax[1].set_title('Costmap')
            ax[2].imshow(self.no_walls)
            ax[2].set_title('dilated costmap')

        if point is not None:
            plt.scatter([point[0]], [point[1]])

        plt.show()

    def save_map(self):
        # same as show_map, but saves the map as a file
        if not self.cost_map is None:
            map1 = self.map_arr
            map2 = self.cost_map

            fig, ax = plt.subplots(1, 3, figsize=(15,5))
            ax[0].imshow(map1)
            ax[0].set_title('Map Array')
            ax[1].imshow(map2)
            ax[1].set_title('Costmap')
            ax[2].imshow(self.no_walls)
            ax[2].set_title('dilated costmap')
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig("final_plot_" + current_time + ".png")
        plt.show()

    def subtract_maps(self, map1, map2):
        result = map1.copy()
        result[(map1 > 70) & (map2 == 100)] = -1
        return result

    def dilate_walls(self, map, W):
        wall_mask = (map == 100).astype(np.uint8)
        kernel = np.ones((W, W), np.uint8)
        dilated_walls = cv2.dilate(wall_mask, kernel, iterations=1)
        result = map.copy()
        result[dilated_walls == 1] = 100
        return result

    def init_costmap_callback(self, msg):
        print('init_costmap_callback')  # For the student to understand
        self.shape = msg.info.height, msg.info.width
        self.cost_map = np.array(msg.data).reshape(self.shape)


    def costmap_callback_update(self, msg):
        shape = msg.height, msg.width
        data = np.array(msg.data).reshape(shape)
        self.cost_map[msg.y:msg.y + shape[0], msg.x: msg.x + shape[1]] = data

        tmp_map = np.copy(self.map_arr)
        tmp_map = self.dilate_walls(tmp_map, 6)
        self.no_walls = self.subtract_maps(self.cost_map, tmp_map)

        if self.number_of_updates % 1e2 == 0:
            self.show_map()
        self.number_of_updates += 1

    def run(self, obj_cen, time_limit):

        print("The initial position is {}".format(self.initial_position))
        planner = GreedyPlanner(self.initial_position)

        sorted_centers = planner.plan(obj_cen)
        print("Will go to {}"%sorted_centers)
        for point in sorted_centers:
            print("about to go to {}".format(point))
            # move the robot to the point
            self.movebase_replan(point, 0)

        #save the final map as a file after all POI are inspected
        self.save_map()

    def movebase_replan(self, point, mode):
        # Creates a new goal with the MoveBaseGoal constructor
        print("Replan started")
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        #go to the [x,y] coordinates of the point
        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]
        # No rotation of the mobile base frame w.r.t. map frame
        goal.target_pose.pose.orientation.w = 1.0

        # Sends the goal to the action server.
        self.client.send_goal(goal)
        rospy.loginfo("New goal command received!")

        while not rospy.is_shutdown():
            if self.obstacle_detected(point) is True:
                print("self position is {}, point is {}".format(self.position, point))
                square = self.square_vertices(self.position, np.array(point))
                print("square is {}, point is {}".format(square, point))
                for corner in square:
                    print("going to corner {}, om the map {}".format(corner,self.position_to_map(corner)))
                    self.movebase_goal(corner)
                break

    def movebase_goal(self, point):
        # Creates a new goal with the MoveBaseGoal constructor
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        #go to the [x,y] coordinates of the point
        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]
        # No rotation of the mobile base frame w.r.t. map frame
        goal.target_pose.pose.orientation.w = 1.0

        # Sends the goal to the action server.
        self.client.send_goal(goal)
        rospy.loginfo("New goal command received!")

        # Waits for the server to finish performing the action.
        wait = self.client.wait_for_result()
        # If the result doesn't arrive, assume the Server is not available
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
        # Result of executing the action
            print("The result is {}".format(self.client.get_result()))
            #return self.client.get_result()


    def obstacle_detected(self, point):
        time.sleep(0.3)
        start = self.position_to_map(self.position)
        goal = self.position_to_map(point)

        current_location = np.round(start).astype(int)
        goal_location = np.round(goal).astype(int)
        # print("current_location is {}, goal_location is {}".format(current_location, goal_location))

        path = self.bresenham2(current_location, goal_location)
        # print("The path is {}".format(path))

        result = self.check_line_collision(path)

        return result

    def bresenham2(self, start, end):
        #accepts 2 points on a plane and returns a discretized straight path from start to end
        start = np.array(start)
        end = np.array(end)
        path = []
        steep = np.abs(end[1] - start[1]) > np.abs(end[0] - start[0])
        if steep:
            start = start[::-1]
            end = end[::-1]
        if start[0] > end[0]:
            start, end = end, start
        delta = np.abs(end - start)
        error = 0
        delta_err = delta[1] / float(delta[0])
        x = start[1]
        for y in np.arange(start[0], end[0]+1, dtype=int):
            if steep:
                path.append((x, y))
            else:
                path.append((y, x))
            error += delta_err
            if error >= 0.5:
                x += np.sign(end[1] - start[1])
                error -= 1
        return path


    def square_vertices(self, corner, center):
        #receives p2 point on the edge and p1 center of a square and returns 4 points forming a square around the center.
        x1, y1 = corner
        x2, y2 = center
        perp = np.cross([x2 - x1, y2 - y1, 0], [0, 0, 1])[:2]
        diag = np.array([x2 - x1, y2 - y1])

        p2 = center + perp
        p3 = center + diag
        p4 = center - perp

        return np.array([
            p2,p3,p4,corner
        ])

    def get_surrounding_points(self, point, center):
        direction = point - center
        distance = np.linalg.norm(direction)
        unit_vector = direction / distance
        rot_matrix = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                               [np.sin(np.pi / 2), np.cos(np.pi / 2)]])
        result = [center + distance * unit_vector]
        for i in range(3):
            unit_vector = np.matmul(rot_matrix, unit_vector)
            result.append(center + distance * unit_vector)
        return np.array(result)


    def euclidean_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def check_line_collision(self, line):
        #accepts an array of points and compares them to the map to see if they overlap. returns the set of overlaping points.
        # print("line length is {}".format(len(line)))
        for i, point in enumerate(line):
            if i == self.replan_distance:
                break
            y, x = point
            #print("no_walls x,y is {},{} value is {}".format(x,y, self.no_walls[x][y]))
            #print("point x,y is {},{} and the costmap value is {}".format(x,y, self.cost_map[x][y]))
            # print("pint x,y is {},{} and the costmap value is {}".format(x,y, self.only_obstacles[x][y]))
            # print("pint x,y is {},{} and the costmap value is {}".format(x,y, self.cost_map[x][y]))
            if self.no_walls[x][y] > 89 and i>(len(line)/3):
                print("point y,x is {},{} and the costmap value is {}, no_walls value is {}, line length is {}".format(x,y, self.cost_map[x][y], self.no_walls[x][y], len(line)))
                return True
        return False

class GreedyPlanner:
    def __init__(self, initial_position):
        self.initial_position = initial_position
        print("self.position: x={}, y={}".format(self.initial_position[0], self.initial_position[1]))

    def plan(self, coords):
        # We use Greedy TSP to plan the path
        num_points = len(coords)
        closest, closest_index = self.closest_point(self.initial_position, coords)
        path = [closest]
        print("closest point is x={}, num={}".format(closest, closest_index))
        remaining_points = np.delete(coords, closest_index, axis=0)
        for i in range(num_points - 1):
            closest, closest_index = self.closest_point(closest, remaining_points)
            print("closest point is x={}, num={}".format(closest, closest_index))
            remaining_points = np.delete(remaining_points, closest_index, axis=0)
            path.append(closest)
        path = np.array(path)
        return path

    def closest_point(self, start, points):
        distances = np.linalg.norm(points - start, axis=1)
        closest_index = np.argmin(distances)
        closest = points[closest_index]
        return closest, closest_index


# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':

    rospy.init_node('assignment2')

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--centers_list",
        nargs="*",
        type=float,
        default=[],
    )
    CLI.add_argument(
        "--time",
        type=float,
        default=2.0,
    )
    args = CLI.parse_args()
    flat_centers = args.centers_list
    time_limit = args.time
    # print(type(time_limit))
    centers = []
    for i in range(len(flat_centers) // 2):
        centers.append([flat_centers[2*i], flat_centers[2*i+1]])

    gcm_client = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer")
    gcm_client.update_configuration({"inflation_radius": 0.3})
    lcm_client = dynamic_reconfigure.client.Client("/move_base/local_costmap/inflation_layer")
    lcm_client.update_configuration({"inflation_radius": 0.3})

    start_time = time.time()
    tb3 = TurtleBot()
    print("tb3 created")
    print("Mapping task started, time limit is {} minutes".format(time_limit))
    tb3.run(centers, time_limit)
    second_measurement = time.time()
    total_time = second_measurement - start_time
    minutes, seconds = divmod(total_time, 60)
    print("Mapping task acheived, total time spent is {:.0f} minutes and {:.2f} seconds".format(minutes, seconds))
