import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SphereEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'sphere1.xml', 4)
        utils.EzPickle.__init__(self)
    
    def _step(self, a):
        ctrl_cost_coeff = 0.0001
        vec = self.get_body_com("ball")-self.get_body_com("wall-up")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        xposbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.data.qpos[0,0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        #reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        #reward = reward_fwd + reward_ctrl
        reward = reward_dist + reward_ctrl + reward_fwd
        ob = self._get_obs()
        return ob, reward, False, dict(reward_dist = reward_dist, reward_ctrl=reward_ctrl, reward_fwd = reward_fwd)
    
    def _get_obs(self):
        """qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        print("qpos:")
        print(qpos)"""
        vec = self.get_body_com("ball")
        print("vec:")
        print(vec)
        return self.get_body_com("ball")
    
    def reset_model(self):
        """self.set_state(
            self.init_qpos + np.random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + np.random.uniform(low=-.1, high=.1, size=self.model.nv)
        )"""
        return self._get_obs()

