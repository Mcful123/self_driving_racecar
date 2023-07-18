import carla
import time
import numpy as np
from math import degrees,radians
import math
from statistics import NormalDist
from itertools import tee
from scipy.spatial import KDTree
import cv2
import matplotlib.pyplot as plt
import heapq

"""
Helper function to find middle points of road
"""
def mid_points(b_pts):
    mids = [None] * len(b_pts[0])
    for i in range(len(mids)):
        left = b_pts[0][i].transform.location
        right = b_pts[1][i].transform.location
        print(right)
        print(left)
        mids[i] = (abs(left.x - right.x)/2, abs(left.y - right.y)/2) 
    return mids

def abs_min(args):
    i = np.argmin(np.abs(args))
    return args[i]

def angleDiff(x, y):
    if x < 0:
        x = 2*np.pi + x
    if y < 0:
        y = 2*np.pi + y
    return (x - y + 3*np.pi) % (2*np.pi) - np.pi
    #return (x - y) % (2*np.pi)
    #return np.arctan2(np.sin(x-y), np.cos(x-y))

def vel_scale(steering, max_vel):
    return min(max_vel, np.sqrt(35*abs(1/np.tan(steering/5))))

def get_mean_waypoints(boundry):
    mean_waypoints = []

    left_x_coords = np.zeros(len(boundry[0]))
    left_y_coords = np.zeros(len(boundry[0]))
    left_t_coords = np.zeros(len(boundry[0]))
    right_x_coords = np.zeros(len(boundry[0]))
    right_y_coords = np.zeros(len(boundry[0]))
    right_t_coords = np.zeros(len(boundry[0]))

    for i in range(len(boundry[0])):
        left_x_coords[i] = boundry[0][i].transform.location.x
        left_y_coords[i] = boundry[0][i].transform.location.y
        left_t_coords[i] = radians(boundry[0][i].transform.rotation.yaw)
        right_x_coords[i] = boundry[1][i].transform.location.x
        right_y_coords[i] = boundry[1][i].transform.location.y
        right_t_coords[i] = radians(boundry[1][i].transform.rotation.yaw)

        mean_x = (left_x_coords[i] + right_x_coords[i])/2
        mean_y = (left_y_coords[i] + right_y_coords[i])/2
        mean_t = (left_t_coords[i] + right_t_coords[i])/2

        dist_left_right = np.sqrt( (left_x_coords[i] - right_x_coords[i])**2 + (left_y_coords[i] - right_y_coords[i])**2)
        mean_waypoints.append((mean_x, mean_y, mean_t, dist_left_right))

    return mean_waypoints

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)



# plt.axis([0,200,0,100])
# plt.ion()
# plt.show(block=False)


class Agent:
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.map_loaded = False
        self.all_mean_points = []
        self.last_plan_waypoint_idx = 0
        self.prev_path = []
        try:
            self.all_mean_points = np.genfromtxt('map.csv', delimiter=',')
            self.map_loaded = True
            self.track_kd = KDTree(self.all_mean_points[:,:2])
            # Pre compute arc_lengths for speedup
            self.arc_lengths = np.zeros(len(self.all_mean_points))
            for i, p in enumerate(pairwise(self.all_mean_points)):
                p1 = p[0]
                p2 = p[1]
                self.arc_lengths[i] = np.sqrt((p1[0] - p2[0])**2 + (p1[0] - p2[0])**2)
        except OSError:
            self.map_loaded = False
            self.all_mean_points = []

    def run_step_map(self, filtered_obstacles, waypoints, vel, transform, boundary):
        print("Reach Customized Agent")
        start = time.time()

        control = carla.VehicleControl()

        x_current = transform.location.x
        y_current = transform.location.y

        theta_current = transform.rotation.yaw * np.pi / 180  # radians
        v_x_current = vel.x
        v_y_current = vel.y
        vel_current = (vel.x ** 2 + vel.y ** 2) ** 0.5

        x_current_waypoint = waypoints[0][0]
        y_current_waypoint = waypoints[0][1]

        #x_next_waypoint = waypoints[1][0]
        #y_next_waypoint = waypoints[1][1]

        #x_diff = x_next_waypoint - x_current_waypoint
        #y_diff = y_next_waypoint - y_current_waypoint
       
        mean_points = get_mean_waypoints(boundary)
        

        for mp in mean_points[:1]:
            if mp not in self.all_mean_points:
                self.all_mean_points.append(mp)
                print("Mapping")
                np_all = np.array(self.all_mean_points)
                np.savetxt('map.csv', np_all, fmt="%f", delimiter=",")
        #print(all_mean_points)
   

        theta_next_waypoint = mean_points[0][2] * 0.25
        theta_next_waypoint += mean_points[1][2] * 0.7
        theta_next_waypoint += mean_points[2][2] * 0.05

        theta_future_waypoint = mean_points[14][2] * 0.25
        theta_future_waypoint += mean_points[15][2] * 0.25
        theta_future_waypoint += mean_points[16][2] * 0.125
        theta_future_waypoint += mean_points[17][2] * 0.125
        theta_future_waypoint += mean_points[18][2] * 0.125
        theta_future_waypoint += mean_points[19][2] * 0.125

        error_x = np.cos(theta_next_waypoint) * (x_current_waypoint - x_current) + np.sin(
            theta_next_waypoint
        ) * (y_current_waypoint - y_current)
        error_y = np.sin(theta_next_waypoint) * (-1) * (
            x_current_waypoint - x_current
        ) + np.cos(theta_next_waypoint) * (y_current_waypoint - y_current) 
        
        error_theta = angleDiff(theta_next_waypoint, theta_current)

        fu_error_theta = angleDiff(theta_future_waypoint, radians(boundary[0][0].transform.rotation.yaw))
        error_v = max(8, vel_scale(fu_error_theta, 20)) - vel_current  # TODO v_ref hard code for now

        lateral_offset_pre_turn = 0
        if abs(fu_error_theta) > 0.25:
            lateral_offset_pre_turn = min(1.5, max(-1.5, fu_error_theta * 2.0))
        error_y += lateral_offset_pre_turn
        # print(f"{error_y}, {lateral_offset_pre_turn}")
        # print(f"{error_y}, {lateral_offset_pre_turn}")

        kx = 0.1
        ky = 0.12
        kv = 0.5
        ktheta = 1.3

        K = np.array([[kx, 0, 0, kv], [0, ky, ktheta, 0]])

        error = np.array([[error_x], [error_y], [error_theta], [error_v]])
        u = K @ error

        normalized_steer = 2 * ((u[1][0] + np.pi) / (2 * np.pi)) - 1

        control.steer = normalized_steer  # u[1]
            
        control.throttle = min(1.0, max(0,0.5 * error_v ))
        if error_v < 0:
            control.brake = min(1.0, -0.1 * error_v)

        end = time.time()
        print(f"Execute time: {end - start}")

        return control

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Execute one step of navigation.

        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints
            - Type:         List[[x,y,z], ...]
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D
            - Description:  Ego's current velocity in (x, y, z) in m/s
        transform
            - Type:         carla.Transform
            - Description:  Ego's current transform
        boundary
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.

        Return: carla.VehicleControl()
        """
        # Actions to take during each simulation step
        # Feel Free to use carla API; however, since we already provide info to you, using API will only add to your delay time
        # Currently the timeout is set to 10s

        if not self.map_loaded:
            return self.run_step_map(filtered_obstacles, waypoints, vel, transform, boundary)

        print("Reach Customized Agent")
        start = time.time()

        control = carla.VehicleControl()

        x_current = transform.location.x
        y_current = transform.location.y

        theta_current = transform.rotation.yaw * np.pi / 180  # radians
        v_x_current = vel.x
        v_y_current = vel.y
        vel_current = (vel.x ** 2 + vel.y ** 2) ** 0.5

        x_current_waypoint = waypoints[0][0]
        y_current_waypoint = waypoints[0][1]

        mean_points = get_mean_waypoints(boundary)

        car_closest_waypoint = None
        car_closest_waypoint_idx = 0
        _, car_closest_waypoint_idx = self.track_kd.query((x_current, y_current), 1)
        car_closest_waypoint = self.all_mean_points[car_closest_waypoint_idx]
        car_lat = -np.sin(car_closest_waypoint[2]) * (x_current - car_closest_waypoint[0]) + np.cos(car_closest_waypoint[2]) * (y_current - car_closest_waypoint[1])
        print(f"Car Lateral: {car_lat}")

        closest_obs_dist_x = 1000
        closest_obs_dist_y = 1000
        obstacle_pos = []
        for o in filtered_obstacles:
            if "vehicle" in o.type_id or "ped" in o.type_id:
                # o.destroy()
                #print(f"{o.type_id}, {o.get_transform().location.y}")

                location = o.get_location()
                
                # find closest waypoint
                closest_waypoint = None
                closest_waypoint_idx = 0
                _, closest_waypoint_idx = self.track_kd.query((location.x, location.y), 1)
                closest_waypoint = self.all_mean_points[closest_waypoint_idx]
                
                # Dist x is the distance along the track between the car and the obstacle
                if closest_waypoint_idx >= car_closest_waypoint_idx:
                    dist_x = np.sum(self.arc_lengths[car_closest_waypoint_idx:closest_waypoint_idx])
                    if dist_x < closest_obs_dist_x:
                        closest_obs_dist_x = dist_x
                else:
                    dist_x = -np.sum(self.arc_lengths[closest_waypoint_idx:car_closest_waypoint_idx])
                # Dist y is the distance across the track
                dist_y = -np.sin(closest_waypoint[2]) * (location.x - closest_waypoint[0]) + np.cos(closest_waypoint[2]) * (location.y - closest_waypoint[1])
                if dist_y < closest_obs_dist_y:
                        closest_obs_dist_y = dist_y
                obstacle_pos.append((dist_x, dist_y, o.type_id))
                #print(f"{o.type_id}, {dist_x}, {dist_y}, {closest_waypoint[:2]}")

        obstacle_map = np.zeros((100, 200)) # each x is 2 meters, each y is 0.05 meters
        for ob in obstacle_pos:
            if ob[0] < 48 and ob[0] > 0 and ob[1] < 5 and ob[1] > -5:
                x_inflation = 3
                y_inflation = 3.5
                lower_x = max(0,  np.floor( (ob[0]- x_inflation) * 0.5 ))
                upper_x = min(100, np.ceil( (ob[0]+ x_inflation) * 0.5 ))
                lower_y = (max(-100, np.floor( (ob[1] - y_inflation)* 20 ))) + 100
                upper_y = (min(100,  np.ceil( (ob[1] + y_inflation)* 20 ))) + 100
                # print(f"{ob}, {lower_x}, {upper_x}, {lower_y}, {upper_y}")

                obstacle_map[int(lower_x): int(upper_x), int(lower_y): int(upper_y)] = 1
        #print(obstacle_map)
        # plt.imshow(obstacle_map, cmap=plt.cm.gray)  # use appropriate colormap here
        # plt.draw()
        # plt.pause(0.001)
        # plt.clf()

        theta_next_waypoint = mean_points[2][2] * 0.25
        theta_next_waypoint += mean_points[3][2] * 0.7
        theta_next_waypoint += mean_points[4][2] * 0.05

        theta_future_waypoint = mean_points[14][2] * 0.25
        theta_future_waypoint += mean_points[15][2] * 0.25
        theta_future_waypoint += mean_points[16][2] * 0.125
        theta_future_waypoint += mean_points[17][2] * 0.125
        theta_future_waypoint += mean_points[18][2] * 0.125
        theta_future_waypoint += mean_points[19][2] * 0.125

        error_x = np.cos(theta_next_waypoint) * (x_current_waypoint - x_current) + np.sin(
            theta_next_waypoint
        ) * (y_current_waypoint - y_current)
        error_y = np.sin(theta_next_waypoint) * (-1) * (
            x_current_waypoint - x_current
        ) + np.cos(theta_next_waypoint) * (y_current_waypoint - y_current) 
        
        fu_error_theta = angleDiff(theta_future_waypoint, radians(boundary[0][0].transform.rotation.yaw))


        dist_to_next_turn = -1
        dir_next_turn = 1
        lateral_offset_pre_turn = 0.0
        normal = NormalDist(mu=0, sigma=10)
        for i in range(car_closest_waypoint_idx, car_closest_waypoint_idx+25):
            idx = (i+1) % len(self.all_mean_points)
            future_waypoint = self.all_mean_points[idx]
            current_turning_angle = angleDiff(car_closest_waypoint[2], self.all_mean_points[car_closest_waypoint_idx-5][2])
            if abs(future_waypoint[2] - car_closest_waypoint[2]) > 0.25 or abs(current_turning_angle) > 0.25:
                dist_to_turn = np.sum(self.arc_lengths[car_closest_waypoint_idx:idx])
                if dist_to_next_turn == -1:
                    dist_to_next_turn = dist_to_turn
                diff = angleDiff(future_waypoint[2], car_closest_waypoint[2])
                if diff < 0:
                    dir_next_turn = -1
                weight = abs(1 * diff + 1/(5*dist_to_turn+1)) - 0.1 * current_turning_angle
                lateral_offset_pre_turn += min(1.0, max(-1.0, 0.05*dir_next_turn*weight))
        
        lateral_offset_pre_turn = min(2, max(-2, lateral_offset_pre_turn))
        # print(f"{lateral_offset_pre_turn}")
        #if abs(fu_error_theta) > 0.25:
        #    lateral_offset_pre_turn = min(1.5, max(-1.5, fu_error_theta * 2.0))
        # error_y += lateral_offset_pre_turn
        # print(f"{error_y}, {lateral_offset_pre_turn}")

        error_theta = angleDiff(theta_next_waypoint, theta_current)

        #if self.last_plan_waypoint[0] == 0 or np.not_equal(self.last_plan_waypoint, car_closest_waypoint).all():
        path = astar(obstacle_map, int(car_lat*20) + 100, int(lateral_offset_pre_turn*20)+100, self.prev_path, error_theta)
        self.prev_path = path
        print(path)
        if path is not None:
            np_path = np.asarray(path)
            self.y_plan = (np_path[:,1] -100)/20
            self.last_plan_waypoint_idx = car_closest_waypoint_idx
        else:
            if np.sum(self.arc_lengths[self.last_plan_waypoint_idx:car_closest_waypoint_idx]) > 2:
                if len(self.y_plan) > 5:
                    self.y_plan = np.append(self.y_plan[1:], self.y_plan[-1])
                    # print(f"APPENDED ARR YPLAN: {self.y_plan}")
                #else:
                #   self.y_plan = [0] * 20

            #print(f"{car_closest_waypoint}, {self.last_plan_waypoint}")
        diff_y = self.y_plan - car_lat
        error_y += diff_y[2] * 0.15
        error_y = diff_y[3] * 0.3
        error_y += diff_y[4] * 0.45
        error_y += diff_y[5] * 0.1

        print(f"Closest obs {closest_obs_dist_x}")
        vel = 0
        if path is not None:
            if dist_to_next_turn == -1:
                dist_to_next_turn = 1000
            dist = min(dist_to_next_turn, closest_obs_dist_x)
            if closest_obs_dist_x < 3 * vel_current:
                vel = max(8, vel_scale(fu_error_theta, 20))
            elif dist < 5 * vel_current:
                vel = max(8, vel_scale(fu_error_theta, 30))
            elif dist < 10 * vel_current:
                vel = max(8, vel_scale(fu_error_theta, 40))
            else:
                vel = max(8, vel_scale(fu_error_theta, 80))
            print(f"vel:{vel}")
        else:
            if closest_obs_dist_x > 3:
                vel = min(10, 10 * (closest_obs_dist_x-3))
            else:
                vel = 0
        vel = vel * min(1, (error_y/3 + 1))
        # vel = max(8, vel_scale(fu_error_theta, 15))
        error_v = vel - vel_current  # TODO v_ref hard code for now

        kx = 0.1
        ky = 0.2
        kv = 0.5
        ktheta = 1.5

        K = np.array([[kx, 0, 0, kv], [0, ky, ktheta, 0]])

        error = np.array([[error_x], [error_y], [error_theta], [error_v]])
        u = K @ error   


        print(error_y)
        # Stanley
        k_e = 0.1
        cross_track_theta = np.arctan(k_e*error_y / (vel))
        # print(cross_track_theta)
        u[1][0] = 1 * error_theta + (5.5 * cross_track_theta) + 0.3 * error_y + 1.8 * np.sin(error_theta)

        # breakpoint()
        normalized_steer = 2 * ((u[1][0] + np.pi) / (2 * np.pi)) - 1

        control.steer = normalized_steer  # u[1]

        control.throttle = min(1.0, max(0,0.5 * error_v ))
        if error_v < 0:
            if error_v < 10:
                control.brake = min(1.0, -0.1 * error_v)
            else:
                control.brake = min(1.0, -0.25 * error_v)

        end = time.time()
        print(f"Execute time: {end - start}")

        return control

def neighbor_cost(current, next, obstacle_map, lat_offset, prev_path):
    cost = current[0]
    cost += 1 + abs((np.arctan((current[2] - next[2])*0.05)/2.0) - current[3])
    cost += 0.1 * abs(next[2]- lat_offset) ** 2

    # if prev_path is not None and len(prev_path) > 0:
    #     cost += 0.05 * abs(next[2] - prev_path[10][1])
    # cost += 0.1 * abs(next[2] - 100)# Center is better
    # if prev_path is not None and len(prev_path) > 0:
    #     # Staying on the same path is better
    #     cost += 0.05 * prev_path[current[1]][1] - current[2]
    #     cost += 0.05 * prev_path[next[1]][1] - next[2]
    # for i, col in enumerate(obstacle_map[current[1]]):
    #     # print(col)
    #     if col != 0:
    #         cost -= 0.001 * abs(next[2] - i) ** 2
    if np.all(obstacle_map[:, next[2]] != 0):
        cost += 10
    # if next[2] < 10 or next[2] > 190:
    #     cost += 40


    # obstacle TODO
    # if obstacle_map[next[1]][next[2]] == 1:
    #     cost += 100

    return cost

def astar(obstacle_map, current_y_idx, lat_offset, prev_path, current_theta):
    obstacle_map_x = 100
    obstacle_map_y = 200
    path = []
    # return 0

    current_y_idx = max(0, min(200, current_y_idx))

    # State tuple = cost from start, x, y, theta heading within road
    # in this case, Y is distance ahead from the car perspective
    starting_state = (0, 0, current_y_idx, current_theta)
    visited_states = {(0, current_y_idx): (None, 0)}
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    while len(frontier) > 0:
        current = heapq.heappop(frontier)
        if current[1] == 20: # and abs(current[2]-lat_offset) <= 20 # y reaches max, finish
            # print("Found it!")
            path =  backtrack(visited_states, current)


            # searched = np.zeros((obstacle_map.shape[0], obstacle_map_y, 3))
            # # print(frontier)
            # # print(len(visited_states))
            # for key in visited_states:
            #     searched[key[0], key[1]] = [1, 1, 1]
            #     # print("hi")
            
            # for p in path:
            #     searched[p[0], p[1]] = [1, 0, 0]

            # print(searched[5])
            # plt.imshow(searched)  # use appropriate colormap here
            # plt.draw()
            # plt.pause(0.001)
            # plt.clf()

            return path

        neighbors = []

        # start = time.time()
        # Next row, close to current y
        for i in range(0, obstacle_map_y, 2):
            if current[1] <= 25 and obstacle_map[current[1]+1, i] == 0:
                #if abs(i-current[2]) < 10: # number of y steps to check in the next row
                if abs(i-current[2]) < 10: # number of y steps to check in the next row
                    n = (0, current[1]+1, i)
                    cost = neighbor_cost(current, n, obstacle_map, lat_offset, prev_path)
                    costly_n = (cost, current[1]+1, i, np.arctan((current[2] - n[2])*0.05)/2.0)
                    neighbors.append(costly_n)
        # end = time.time()
        # print(neighbors)
        # print(f"Execute time of creating neighbors: {end - start}")

        for n in neighbors:
            prev = visited_states.get( (n[1], n[2]) )
            if prev is None or prev[1] > n[0]:
                heapq.heappush(frontier, n)
                visited_states[(n[1], n[2])] = ((current[1], current[2]), n[0])

    searched = np.zeros_like(obstacle_map)
    # print(frontier)
    # print(len(visited_states))
    for key in visited_states:
        searched[key[0], key[1]] = 1
        # print("hi")
    # print(searched[5])
    # plt.imshow(searched, cmap=plt.cm.gray)  # use appropriate colormap here
    # plt.draw()
    # plt.pause(0.001)
    # plt.clf()

    return None

def backtrack(visited_states, current_state):
    path = []
    curr_state = (current_state[1], current_state[2]) # remove the 0 idx
    while curr_state is not None:
        path.append(curr_state)
        curr_state = visited_states[curr_state][0]

    path.reverse()
    return path

if __name__ == "__main__":
    test_points = [ (-1, 1, -2) , (-10, 160, -30)]
    for tp in test_points:
        result = angleDiff(math.radians(tp[0]), math.radians(tp[1]))
        # print(f" {tp[0]} - {tp[1]} = {degrees(result)}")
        assert(math.isclose(result, math.radians(tp[2]), rel_tol=1e-4))
