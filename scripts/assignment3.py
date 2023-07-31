#!/usr/bin/env python

from click import get_app_dir
import rospy
import yaml
import os
import argparse
import numpy as np

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseWithCovarianceStamped
import dynamic_reconfigure.client

from itertools import combinations, permutations, product

from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetMap, GetPlan

import actionlib
from MRS_236609.srv import ActionReq
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from costmap_listen_and_update import CostmapUpdater
from MRS_236609.srv import GetCostmap

import math
import time


CUBE_EDGE = 0.5



class TurtleBot:
    def __init__(self):
        self.initial_position = None
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, callback=self.set_initial_position)
        self.time = None
        self.start_time = None

        self.ws = None
        self.tasks = None
        self.tasks_list = {}

        self.distances = {}
        self.distances_cost = {}

        rospy.Service('/initial_costmap', GetCostmap, self.get_costmap)
        self.cmu = CostmapUpdater()

        print("Waiting for an initial position...")
        while self.initial_position is None:
            continue
        print("The initial position is {}".format(self.initial_position))

    def set_initial_position(self, msg):
        initial_pose = msg.pose.pose
        self.initial_position = np.array([initial_pose.position.x, initial_pose.position.y])
    
    def get_costmap(self, req):
        return self.cmu.initial_msg

    def run(self, ws, tasks, time_limit):
        self.start_time = time.time()
        self.time = time_limit*60
        print("Time is {}".format(self.time))
        self.ws = ws
        self.num_ws = len(ws)

        self.client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        self.client.wait_for_server()


        # ==== Printing the tasks =======

        print("Tasks are:", tasks)
        for w, val in ws.items():
            print(w + ' center is at ' + str(val.location) + ' and its affordance center is at ' + str(
                val.affordance_center))
            print(w + ' possible tasks are ' + str(val.tasks))


        # Create new dictionary of tasks that can hold actions and paths in each task
        for chain_key, reward in tasks.items():
            self.tasks_list[chain_key] = Task(reward)

        # calculate distances and costs between all ws
        self.distances, self.distances_cost = self.calculate_distance_dict(self.ws)

        print("Distances are ", self.distances)
        print("Distances cost are ", self.distances_cost)

        # Check what tasks are executable and update self.tasks_list the dictionary where the key is the task and actions is the list of possible ways to execute the task
        self.check_all_tasks(tasks)

        # Search for the optimal path by taking all possible combinations in the dictionary cut by timeout

        max_time = self.time

        comb = self.combine_actions_with_time(self.tasks_list, max_time)
        # print("All possible combinations are {}".format(comb))

        max_reward = max([item[1][0] for item in comb])
        elements_with_max_reward = [item for item in comb if item[1][0] == max_reward]

        # Find the element with the smallest total_cost among elements with maximum total_reward
        best_comb = min(elements_with_max_reward, key=lambda x: x[1][2])
        print("Result: Best combination is {}".format(best_comb))
        
        # Execute that path with move_base
        #we have list of ws and list of actions, try all actions in order until fail, then move to the next ws

        self.execute_action_sequence(best_comb[0])

        current_time = time.time()
        time_used = current_time - self.start_time
        print("The plan is succesfuly executed, the robot is now at the goal, total reward is {}, estimated time is {}, total execution time is {}".format(best_comb[1][0], best_comb[1][1], time_used))

        # ===========================


    # ===========================================================================================================
    # Functions for calculating all possible path costs
    # ===========================================================================================================

    def get_plan(self, start_coord, goal):
        """
        Generates a path plan for a robot in a ROS-based environment from a given start coordinate to a goal coordinate.

        :param start_coord: A tuple representing the starting coordinate (x, y) of the robot.
        :param goal: A tuple representing the goal coordinate (x, y) the robot should reach.
        :return: A Path object containing the computed path plan from the starting coordinate to the goal coordinate.
        """
        start = PoseStamped()
        start.header.seq = 0
        start.header.frame_id = "map"
        start.header.stamp = rospy.Time(0)
        start.pose.position.x = start_coord[0]
        start.pose.position.y = start_coord[1]

        Goal = PoseStamped()
        Goal.header.seq = 0
        Goal.header.frame_id = "map"
        Goal.header.stamp = rospy.Time(0)
        Goal.pose.position.x = goal[0]
        Goal.pose.position.y = goal[1]

        gcm_client = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer")
        gcm_client.update_configuration({"inflation_radius": 0.4})
        lcm_client = dynamic_reconfigure.client.Client("/move_base/local_costmap/inflation_layer")
        lcm_client.update_configuration({"inflation_radius": 0.4})       

        make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        new_plan = GetPlan()
        new_plan.start = start
        new_plan.goal = Goal
        plan = make_plan(new_plan.start, new_plan.goal, 1)

        gcm_client = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer")
        gcm_client.update_configuration({"inflation_radius": 0.2})
        lcm_client = dynamic_reconfigure.client.Client("/move_base/local_costmap/inflation_layer")
        lcm_client.update_configuration({"inflation_radius": 0.2})

        return plan

    def get_plan_length(self, plan):
        last_item = plan[0]
        plan_length = 0
        for item in plan:
            plan_length += math.sqrt((item.pose.position.x - last_item.pose.position.x)**2 + (item.pose.position.y - last_item.pose.position.y)**2)
            last_item = item
        return plan_length
    
    def get_plan_cost(self, plan):
        plan_cost = 0
        for item in plan:
            # print("item.pose.position is {}".format(item.pose.position))
            cur_pose = np.array([item.pose.position.x, item.pose.position.y])
            idx_x, idx_y = self.cmu.position_to_map(cur_pose)
            c = self.cmu.cost_map[int(idx_x)][int(idx_y)]
            plan_cost += c
        return plan_cost


    def calculate_distance_dict(self, ws):
        """
        Calculates the matrix of distances between affordance_center's of Workstations.
        :param workstations: A dictionary of Workstation objects.
        :return: A 2D matrix where matrix[start][end] is a distance from start to end.
        """

        # Step 1: Get all possible pairs of Workstations
        ws_keys = ws.keys()
        ws_combinations = list(combinations(ws_keys, 2))

        distances = {}
        distances_cost = {}

        # Distance between (i, i) is 0
        for ws_key in ws_keys:
            distances[(ws_key, ws_key)] = 0
            distances_cost[(ws_key, ws_key)] = 0

            # Calculate Distance between start and all ws
            start_center = self.initial_position
            end_center = ws[ws_key].affordance_center

            # print("start is {}, ws is {}".format(start_center, end_center))

            plan_forward = self.get_plan(start_center, end_center).plan.poses
            distances[('start', ws_key)] = self.get_plan_length(plan_forward)
            cost_forward = self.get_plan_cost(plan_forward)
            distances_cost[('start', ws_key)] = cost_forward


        # Step 2: Calculate the distance between affordance_centers for each pair and save it to the distances dict
        for start, end in ws_combinations:
            start_center = ws[start].affordance_center
            end_center = ws[end].affordance_center

            # print("ws")

            plan_forward = self.get_plan(start_center, end_center).plan.poses
            plan_backward = self.get_plan(end_center, start_center).plan.poses

            distances[(start, end)] = self.get_plan_length(plan_forward)
            distances[(end, start)] = self.get_plan_length(plan_backward)

            cost_forward = self.get_plan_cost(plan_forward)
            cost_backward = self.get_plan_cost(plan_backward)

            distances_cost[(start, end)] = cost_forward
            distances_cost[(end, start)] = cost_backward

        return distances, distances_cost



    # ===========================================================================================================
    # Functions for calculating all possible sequences to complete the task list
    # ===========================================================================================================

    
    def get_combinations_with_order(self, elements, k, prev=None):
        if k == 0:
            return [[]]

        combinations = []
        for element in elements:
            if element != prev:
                for sub_combination in self.get_combinations_with_order(elements, k - 1, element):
                    combinations.append([element] + sub_combination)

        return combinations



    def check_all_tasks(self, tasks):
        """
-----   Checks if all tasks can be performed on the given workstations
        """
        dict = {}
        # Get all chains of ACTs as lists from the tasks
        for chain_key, value in tasks.items():
            task_chain = chain_key.split('->')
            # task_chain = [(chain[i], chain[i+1]) for i in range(len(chain)-1)]
            self.tasks_list[chain_key].update_chain(task_chain)

            # Get all possible workstation sequences of task length
            ws_keys = list(self.ws.keys())
            # print("ws_keys is {}".format(ws_keys))

            # ws_combinations = [list(combination) for combination in permutations(ws_keys, len(task_chain))]

            ws_combinations = self.get_combinations_with_order(ws_keys, len(task_chain))

            # print("ws_combinations is {}".format(ws_combinations))

            # print("Current task_chain is {}".format(task_chain))

            #Check if the task can be performed on those ws sequences
            for ws_chain in ws_combinations:

                # print("Current ws_chain is {}, length is {}".format(ws_chain, len(ws_chain)))
                      
                action_sequence = self.check_one_task(ws_chain, task_chain)

                # print("action_sequence is {}".format(action_sequence))

                # for each task we save all possible combinations of actions to acheive it
                if len(action_sequence) is not 0:
                    self.tasks_list[chain_key].update_actions(action_sequence)

        print("tasks_list is {}".format(self.tasks_list))
        for task_key in self.tasks_list:
            print("task is {}".format(task_key))
            print("task.actions is {}".format(self.tasks_list[task_key].actions))
    

    def check_one_task(self, ws_chain, task_chain):
    
        """
        Checks if task can be performed on len(task_chain) workstations.
        :ws_chain: A sequence of workstations
        :task_chain: A list of activities to perform
        :return: A sequence of actions needed to perform that task (ACT1->PU-A->PL-A->ACT2), empty list if such sequence not exist
        """

        action_sequence = []

        for i, ws_name in enumerate(ws_chain[:-1]):
            ws1 = self.ws[ws_name]
            ws2 = self.ws[ws_chain[i + 1]]

            act1 = task_chain[i]
            act2 = task_chain[i + 1]

            if act1 not in ws1.tasks or act2 not in ws2.tasks:
                return []

            common_tasks = set(ws1.tasks) & set(ws2.tasks)
            pu_pl_pairs = [(t1, t2) for t1 in common_tasks for t2 in common_tasks if t1.startswith('PU-') and t2.startswith('PL-') and t1[3:] == t2[3:]]

            if not pu_pl_pairs:
                return []

            pu, pl = pu_pl_pairs[0]

            # action_sequence.extend([(ws_name, act1), (ws_name, pu), (ws_chain[i + 1], pl), (ws_chain[i + 1], act2)])

            # action_sequence.extend([(ws1, act1), (ws1, pu), (ws2, pl), (ws2, act2)])

            action_sequence.extend([(ws_name, act1), (ws_name, pu), (ws_chain[i + 1], pl)])
            if i == len(ws_chain) - 2:
                action_sequence.append((ws_chain[i + 1], act2))

        return action_sequence

    # ===========================================================================================================
    # Functions for choosing the best possible path
    # ===========================================================================================================

    def combine_actions(self, tasks_list, current_tasks=None, current_actions=None, index=0):
        if current_tasks is None:
            current_tasks = []
        if current_actions is None:
            current_actions = []
        if index == len(tasks_list):
            return [current_actions] if len(current_tasks) == len(tasks_list) else []

        task_key = list(tasks_list.keys())[index]
        task = tasks_list[task_key]
        results = []

        for action_sequence in task.actions:
            new_tasks = list(current_tasks) 
            new_tasks.append(task_key)
            new_actions = list(current_actions)
            new_actions.extend(action_sequence)
            results.extend(self.combine_actions(tasks_list, new_tasks, new_actions, index + 1))

        results.extend(self.combine_actions(tasks_list, current_tasks, current_actions, index + 1))

        return results
    
    def combine_actions_with_time(self, tasks_list, max_time, current_tasks=None, current_actions=None, index=0):
        """
        Recursively generates all possible action sequences based on the given tasks_list and calculates the total
        reward, time, and cost for each valid sequence. A sequence is considered valid if its total time does not
        exceed the specified max_time.

        :param tasks_list: A dictionary of tasks where the key is the task identifier and the value is a Task object.
        :param max_time: A float representing the maximum allowed time for a valid sequence.
        :param current_tasks: (Optional) Recursion parameter. A list of task identifiers for the tasks in the current sequence.
                            Default is an empty list.
        :param current_actions: (Optional) Recursion parameter. A list of tuples representing the current sequence of actions.
                                Default is an empty list.
        :param index: (Optional) Recursion parameter. An integer representing the current index of the task being processed in tasks_list.
                    Default is 0.

        :return: A list of tuples where each tuple contains a valid action sequence and its corresponding total
                reward, time, and cost.
        """
        if current_tasks is None:
            current_tasks = []
        if current_actions is None:
            current_actions = []
        if index == len(tasks_list):
            start_time = 0
            if len(current_actions) > 0:
                # print("current_actions is {}".format(current_actions))
                # print("current_actions[0] is {}".format(current_actions[0]))
                distance = self.distances[('start', current_actions[0][0])]
                start_time = distance / 0.22
                # print("Start time is {}".format(start_time))
                # print("Total time is {}".format(self.calculate_time(current_actions, self.distances)))
            cur_time = self.calculate_time(current_actions, self.distances) + start_time
            if cur_time <= max_time:
                total_reward = sum([tasks_list[task_key].reward for task_key in current_tasks])
                total_cost = self.calculate_cost(current_actions, self.distances_cost)
                res = (total_reward, cur_time, total_cost)
                # print("Start time is {}".format(start_time))
                # print("Current time is {}".format(cur_time))
                # print("Total time is {}".format(self.calculate_time(current_actions, self.distances)))
                return [(current_actions, res)]
                # return [(current_actions, total_reward)] if len(current_tasks) == len(tasks_list) else []
            else:
                return []

        task_key = list(tasks_list.keys())[index]
        task = tasks_list[task_key]
        results = []

        for action_sequence in task.actions:
            new_tasks = list(current_tasks)  
            new_tasks.append(task_key)
            new_actions = list(current_actions)
            new_actions.extend(action_sequence)
            results.extend(self.combine_actions_with_time(tasks_list, max_time, new_tasks, new_actions, index + 1))

        results.extend(self.combine_actions_with_time(tasks_list, max_time, current_tasks, current_actions, index + 1))

        return results
    
    def calculate_time(self, actions_list, distances):
        speed = {'PU': 0.15, 'PL': 0.22, 'ACT': 0.22}
        total_time = 0

        for i in range(len(actions_list) - 1):
            el_1 = actions_list[i]
            el_2 = actions_list[i + 1]
            distance = distances[(el_1[0], el_2[0])]

            if distance is not None:
                if el_1[0].startswith('PU'):
                    action_speed = 0.15
                else:
                    action_speed = 0.22
                time = distance / action_speed
                total_time += time

        return total_time
    
    def calculate_cost(self, actions_list, distances_cost):
        total_cost = 0

        for i in range(len(actions_list) - 1):
            el_1 = actions_list[i]
            el_2 = actions_list[i + 1]
            cost = distances_cost[(el_1[0], el_2[0])]
            total_cost += cost

        return total_cost


    def calculate_path_cost(self, points):
        cost = 0
        for i in range(len(points) - 1):
            cost += self.distance_matrix(points[i], points[i+1])
        return cost


    # ===========================================================================================================
    # Functions for implementing the path
    # ===========================================================================================================


    def execute_action_sequence(self, action_sequence):
        """
        Executes a given action sequence by navigating to each workstation's affordance center and performing the
        specified action. This function communicates with a "/do_action" service to execute the
        actions.

        :param action_sequence: A list of tuples, where each tuple contains a workstation identifier and the
                                corresponding action to perform.

        :return: None
        """
        print("Executing action sequence: {}".format(action_sequence))
        do_action = rospy.ServiceProxy("/do_action", ActionReq)

        for action in action_sequence:
            self.movebase_goal(self.ws[action[0]].affordance_center)
            res = do_action(action[0], action[1])
            print("Action {} result: {}".format(action, res.success))
            assert(res.success == True)


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


# ======================================================================================================================

def analyse_res(msg):
    result = {}
    for line in msg.split("\n"):
        if line:
            parts = line.split(" ")
            key = parts[0]
            x = float(parts[-2])
            y = float(parts[-1])
            result[key] = [x, y]
    return result


class Workstation:
    def __init__(self, location, tasks):
        self.location = location
        self.tasks = tasks
        self.affordance_center = None

    def update_affordance_center(self, new_center):
        self.affordance_center = new_center

class Task:
    def __init__(self, reward):
        self.actions = []
        self.reward = reward
        self.chain = None

    def update_chain(self, task_list):
        self.chain = task_list

    def update_actions(self, action_sequence):
        self.actions.append(action_sequence)


# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    rospy.init_node('assignment3')

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--time",
        type=float,
        default=2.0,
    )
    args = CLI.parse_args()
    time_limit = args.time

    gcm_client = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer")
    gcm_client.update_configuration({"inflation_radius": 0.2})
    lcm_client = dynamic_reconfigure.client.Client("/move_base/local_costmap/inflation_layer")
    lcm_client.update_configuration({"inflation_radius": 0.2})

    ws_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/workstations_config.yaml"
    tasks_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/tasks_config.yaml"
    ws = {}
    with open(ws_file, 'r') as f:
        data = yaml.load(f)
        num_ws = data['num_ws']
        for i in range(num_ws):
            ws['ws' + str(i)] = Workstation(data['ws' + str(i)]['location'], data['ws' + str(i)]['tasks'])

    rospy.wait_for_service('/affordance_service', timeout=5.0)
    service_proxy = rospy.ServiceProxy('/affordance_service', Trigger)
    res = service_proxy()
    aff = analyse_res(res.message)
    for key, val in ws.items():
        val.update_affordance_center(aff[key])

    with open(tasks_file, 'r') as f:
        data = yaml.load(f)
        tasks = data['tasks']

    tb3 = TurtleBot()
    tb3.run(ws, tasks, time_limit)
