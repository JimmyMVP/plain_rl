from torch_rl.training.core import HorizonTrainer, mse_loss
from torch.optim import Adam
from torch_rl.utils import to_tensor as tt
import torch as tor
from collections import deque
from torch_rl.utils import prGreen
import time
import sys
from torch_rl.utils import logger
import numpy as np

def queue_to_array(q):
    q.put(False)
    arr = []
    while True:
        item = q.get()
        if item:
            arr.append(item)
        else:
            break

    return np.asarray(arr)




class AdvantageEstimator(object):

    def __init__(self, env, network, nsteps, gamma, lam):
        self.env = env
        self.network = network
        nenv = 1
        self.obs = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.state = [] if network.recurrent else None
        self.done = False
        self.global_step = 0
        self.episodes = 0
        self.recurrent = network.recurrent

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logpacs = [], [], [], [], [], []
        mb_states = self.state
        epinfos = []
        for _ in range(self.nsteps):
            if mb_states is None:
                actions, values = self.network(tt(self.obs, cuda=False).view(1,-1))
            else:
                state_critic = self.network.lh_val
                state_policy = self.network.lh_pol
                actions, values = self.network(tt(self.obs, cuda=False).view(1,1,-1), use_last_state=True)
                if state_critic is None:
                    state_critic = np.zeros_like(self.network.lh_val.data.numpy())
                    state_policy = np.zeros_like(self.network.lh_pol.data.numpy())
                else:
                    state_critic = state_critic.data.numpy()
                    state_policy = state_policy.data.numpy()

                mb_states.append([state_policy,state_critic])


            logpacs = self.network.logprob(actions)

            mb_obs.append(self.obs.copy().flatten())
            mb_actions.append(actions.data.numpy().flatten())
            mb_values.append(values.detach().data.numpy().flatten())
            mb_logpacs.append(logpacs.data.numpy().flatten())

            mb_dones.append(self.done)

            obs, reward, self.done, infos = self.env.step(actions.data.numpy().flatten())
            self.obs = obs
            self.global_step += 1
            mb_rewards.append(reward)

            if self.done:
                self.episodes+=1
                logger.logkv("episodes", self.episodes)
                self.obs = self.env.reset()
                if self.network.recurrent:
                    self.network.reset() 


        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape(self.nsteps, -1)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape(self.nsteps, -1)
        mb_actions = np.asarray(mb_actions, dtype=np.float32).reshape(self.nsteps, -1)
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape(self.nsteps, -1)
        mb_logpacs = np.asarray(mb_logpacs, dtype=np.float32).reshape(self.nsteps, -1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).reshape(self.nsteps, -1)
        if not mb_states is None:
            #mb_states = np.asarray(mb_states, dtype=np.float32).reshape(self.nsteps, 1, -1)
            action, last_values = self.network(tt(self.obs.reshape(1,1,-1), cuda=False), use_last_state=True)
            mb_states = np.asarray(mb_states, dtype=np.float32)

        else:
            action, last_values = self.network(tt(self.obs.reshape(1,-1), cuda=False))

        action, last_values = action.data.numpy().reshape(-1), last_values.data.numpy().reshape(-1)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.done
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values


        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_logpacs, mb_states

        # obs, returns, masks, actions, values, neglogpacs, states = runner.run()

    def constfn(val):
        def f(_):
            return val

        return f


class GPUPPOTrainer(HorizonTrainer):

    mvavg_reward = deque(maxlen=100)


    def __init__(self, env, network, max_episode_len=500, gamma=.99, lr=3e-4, n_steps=40,
                 epsilon=0.2, optimizer=None, lmda=0.95, ent_coef=0., n_update_steps=10, num_threads=5, n_minibatches=1):
        super(GPUPPOTrainer, self).__init__(env)

        self.n_minibatches = n_minibatches
        self.lr = lr
        self.max_episode_len = max_episode_len
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmda = lmda
        self.optimizer = Adam(network.parameters(), lr=lr) if optimizer is None else optimizer
        self.goal_based = hasattr(env, "goal")
        self.network = network
        self.recurrent = network.recurrent
        self.ent_coef = ent_coef
        self.num_threads = num_threads
        self.n_update_steps = n_update_steps
        self.n_steps = n_steps
        self.advantage_estimator = AdvantageEstimator(env, self.network, n_steps, self.gamma, self.lmda)

    def _horizon_step(self):


        obs, returns, masks, actions, values, logpacs, states = self.advantage_estimator.run() #pylint: disable=E0632
        #Normalize advantages over episodes
        advs = returns - values
        prev_ind = 0
        for ind in np.argwhere(masks == True)[:, 0]:
            episode_advs = advs[prev_ind:ind+1]
            advs[prev_ind:ind+1] = (episode_advs - episode_advs.mean())/(episode_advs.std() + 1e-8)
            prev_ind = ind+1

        episode_advs = advs[prev_ind:-1]
        advs[prev_ind:-1] = (episode_advs - episode_advs.mean())/(episode_advs.std() + 1e-8)
    

        nbatch_train = self.n_steps // self.n_minibatches

        
        if tor.cuda.is_available():
            network = self.network.cuda()
        else:
            network=self.network
        #self.optimizer = Adam(self.network.parameters(), lr=self.lr) 
        if not states is None:
            #store last network state
            lh_val = network.lh_val
            lh_pol = network.lh_pol

        inds = np.arange(self.n_steps)
        for _ in range(self.n_update_steps):
            np.random.shuffle(inds)
            for start in range(0, self.n_steps, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                if states is None: # nonrecurrent version

                    bobs, breturns, bmasks, bactions, bvalues, blogpacs, badvs = map(\
                        lambda arr: arr[mbinds], (obs, returns, masks, actions, values, logpacs, advs))
                    # This introduces bias since the advantages can be normalized over more episodes
                    #advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                    OBS = tt(bobs)
                    A = tt(bactions)
                    ADV = tt(badvs)
                    R = tt(breturns)
                    OLDLOGPAC = tt(blogpacs)
                    OLDVPRED = tt(bvalues)

                    actions_new, v_pred  = self.network(OBS)
                    logpac = self.network.logprob(A)
                    entropy = tor.mean(self.network.entropy())

                    #### Value function loss ####
                    #print(bobs)
                    v_pred_clipped = OLDVPRED + tor.clamp(v_pred - OLDVPRED, -self.epsilon, self.epsilon)
                    v_loss1 = (v_pred - R)**2/2.
                    v_loss2 = (v_pred_clipped - R)**2/2.

                    v_loss = .5 * tor.mean(tor.max(v_loss1, v_loss2))

                    ### Ratio calculation ####
                    # In the baselines implementation these are negative logits, then it is flipped
                    ratio = tor.exp(logpac - OLDLOGPAC)

                    ### Policy gradient calculation ###
                    pg_loss1 = -ADV * ratio
                    pg_loss2 = -ADV * tor.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon)
                    pg_loss = tor.mean(tor.max(pg_loss1, pg_loss2))
                    approxkl = .5 * tor.mean((logpac - OLDLOGPAC)**2)


                    loss = v_loss  + pg_loss + self.ent_coef*entropy

                    #clipfrac = tor.mean((tor.abs(ratio - 1.0) > self.epsilon).type(tor.FloatTensor))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                else:

                    bobs, breturns, bmasks, bactions, bvalues, blogpacs, badvs, bstates = map(\
                        lambda arr: arr[mbinds], (obs, returns, masks, actions, values, logpacs, advs, states))
                    OBS = tt(bobs)
                    A = tt(bactions)
                    ADV = tt(badvs)
                    R = tt(breturns)
                    OLDLOGPAC = tt(blogpacs)
                    OLDVPRED = tt(bvalues)

                    STATES_POLICY = np.asarray([x for x,y in bstates]).squeeze().transpose((1,0,2))
                    STATES_CRITIC = np.asarray([y for x,y in bstates]).squeeze().transpose((1,0,2))

                    self.network.lh_pol = tt(STATES_POLICY)
                    self.network.lh_val = tt(STATES_CRITIC)
                    actions_new, v_pred  = self.network(OBS.view(nbatch_train, 1, -1), use_last_state=True)

                    logpac = self.network.logprob(A)
                    entropy = tor.mean(self.network.entropy())

                    #### Value function loss ####
                    #print(bobs)
                    v_pred_clipped = OLDVPRED + tor.clamp(v_pred - OLDVPRED, -self.epsilon, self.epsilon)
                    v_loss1 = (v_pred - R)**2/2.
                    v_loss2 = (v_pred_clipped - R)**2/2.

                    v_loss = .5 * tor.mean(tor.max(v_loss1, v_loss2))

                    ### Ratio calculation ####
                    # In the baselines implementation these are negative logits, then it is flipped
                    ratio = tor.exp(logpac - OLDLOGPAC)

                    ### Policy gradient calculation ###
                    pg_loss1 = -ADV * ratio
                    pg_loss2 = -ADV * tor.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon)
                    pg_loss = tor.mean(tor.max(pg_loss1, pg_loss2))
                    approxkl = .5 * tor.mean((logpac - OLDLOGPAC)**2)


                    loss = v_loss  + pg_loss + self.ent_coef*entropy

                    #clipfrac = tor.mean((tor.abs(ratio - 1.0) > self.epsilon).type(tor.FloatTensor))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        if not states is None:
            #restore last states for runner
            self.network.lh_pol = lh_pol
            self.network.lh_val = lh_val

        #Push to CPU
        self.network.cpu()
        logger.logkv("siglog", self.network.siglog.data.numpy()[0])
        logger.logkv("pgloss", pg_loss.cpu().data.numpy())
        logger.logkv("vfloss", v_loss.cpu().data.numpy())
        logger.logkv("vfloss", v_loss.cpu().data.numpy())
        logger.logkv("approxkl", approxkl.cpu().data.numpy())
        logger.logkv("pentropy", entropy.cpu().data.numpy())
        logger.dumpkvs()



if __name__ == '__main__':

    from torch_rl.envs.wrappers import *
    import gym
    from gym.wrappers import Monitor
    from torch_rl.models.ppo import ActorCriticPPO, RecurrentActorCriticPPO
    from torch_rl.utils import *
    from torch_rl.utils import logger
    from torch_rl.envs import EnvLogger
    from torch import nn
    import roboschool
    import sys

    logger.configure(clear=False)
    env = EnvLogger(NormalisedActionsWrapper(gym.make("RoboschoolReacher-v1")))
    env = RunningMeanStdNormalize(env)
    print(env.observation_space.shape)


    net = RecurrentActorCriticPPO([env.observation_space.shape[0],32, env.action_space.shape[0]
        ], recurrent_layers=2, recurr_type=nn.LSTM, bidirectional=False)
    net.apply(gauss_init(0, np.sqrt(2)))

    trainer = GPUPPOTrainer(network=net, env=env, n_update_steps=10, 
        n_steps=1024, n_minibatches=1, lmda=.95, gamma=.99, lr=3e-4, epsilon=0.2, ent_coef=0.0)
    trainer.train(horizon=100000, max_episode_len=500)




