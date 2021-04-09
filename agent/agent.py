import pandas as pd
import numpy as np

from pathlib import Path
import pickle
import gym
import warnings
warnings.filterwarnings('ignore')

# how to import or load local files
import os
import sys
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
import gym_cfg
with open(path + "/gym_cfg.py", "r") as f:
    pass

class TestAgent():
    def __init__(self):
        self.now_phase = {}
        self.green_sec = 2
        self.red_sec = 0
        self.max_phase = 8
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}
        self.phase_mv_inc = [(1, 2), (1, 4), (2, 3), (2, 5),
                             (3, 0), (3, 6), (4, 1), (4, 7),
                             (5, 2), (5, 3), (6, 0), (6, 1),
                             (7, 4), (7, 5), (8, 6), (8, 7)]
        self.road_idx_to_arm = {0:"Ni", 1:"Ei", 2:"Si", 3:"Wi",
                                4:"No", 5:"Eo", 6:"So", 7:"Wo"}
        self.lane_idx_to_mv = {0:"L", 1:"T", 2:"R"}
        self.mv_to_phase = {
            "NiL":1, "SiL":1, "NiT":2, "SiT":2,
            "EiL":3,  "WiL":3, "EiT":4, "WiT":4
        }
        self.prev_action = None
        
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
    ################################

    
    def obs_process(self, observations):
        # preprocess observations
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val
        
        # format into pd.dataframe
        obs = []
        for agent_id, agent_obs in observations_for_agent.items():
            sim_step = agent_obs['lane_speed'][0]
            lane_speed = agent_obs['lane_speed'][1:]
            lane_vehicle_num = agent_obs['lane_vehicle_num'][1:]
            assert len(lane_speed) == len(lane_vehicle_num)
            for idx, speed in enumerate(lane_speed):
                road_idx = idx // 3
                lane_idx = idx % 3
                veh_num = lane_vehicle_num[idx]
                obs.append([sim_step, agent_id, road_idx, lane_idx, speed, veh_num])

        obs_df = pd.DataFrame(obs, columns=['sim_step', 'agent_id', 'road_idx', 'lane_idx', 'speed', 'veh_num'])
        
        return obs_df
        
    
    def gen_pressure(self, obs_df):
        
        # formatting obs_df
        obs_df = obs_df[(obs_df.road_idx < 4) & (obs_df.lane_idx < 2)]
        obs_df['road_idx'] = obs_df['road_idx'].replace(self.road_idx_to_arm)
        obs_df['lane_idx'] = obs_df['lane_idx'].replace(self.lane_idx_to_mv)
        obs_df['mv'] = obs_df['road_idx'] + obs_df['lane_idx']
        obs_df['phase'] = obs_df['mv'].replace(self.mv_to_phase)
        
        # define pressure: the number of vehicles on the approach
        pressure = obs_df.pivot_table(index='agent_id',
                                      columns='mv',
                                      values='veh_num',
                                      aggfunc='sum')
        
        return pressure
    
    
    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos

        # observations is returned 'observation' of env.step()
        # info is returned 'info' of env.step()
        observations = obs['observations']
        info = obs['info']
        actions = {}

        # a simple fixtime agent

        # preprocess observations
        obs_df = self.obs_process(observations)
        cur_step = obs_df['sim_step'].unique()[0]
        if (self.prev_action is not None) and (cur_step % self.green_sec > 1):
            return self.prev_action
        
        # get pressure
        pressure = self.gen_pressure(obs_df)

        # get actions
        phase_mv_incidence = np.zeros((4, 8))
        for x, y in self.phase_mv_inc[:8]:
            phase_mv_incidence[x - 1, y] = 1
        
        action = {}
        for i in range(len(pressure)):
            agent_id = pressure.index[i]
            phase_pressures = np.dot(phase_mv_incidence, pressure.iloc[i, :].values)
            max_locs = np.where(phase_pressures == np.max(phase_pressures))[0]
            if np.std(phase_pressures) == 0:
                action[agent_id] = np.random.randint(1, 5, 1)[0]
            elif len(max_locs) > 1:
                loc = np.random.randint(0, len(max_locs), 1)
                action[agent_id] = max_locs[loc][0] + 1
            else:
                agent_action = np.argmax(phase_pressures) + 1
                action[agent_id] = agent_action
        self.prev_action = action
        
        return action

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()

