from torch import nn
from torch_rl.utils import gauss_weights_init

from torch_rl.core import *
import os
import glob

class SaveableModel(object):


    def save(self, path):
        tor.save(self, path)

    @classmethod
    def load(cls, path):
        return tor.load(path)


    @classmethod
    def load_best(cls, path):
        assert os.path.isdir(path)

        best_models = glob.glob(os.path.join(path, "*best*"))

        assert not len(best_models > 1)

        return tor.load(os.path.join(path, best_models[0]))


class NeuralNet(nn.Module, SaveableModel):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def action(self, x):
        pass




class StochasticNeuralNet(NeuralNet):

    def __init__(self):
        super(StochasticNeuralNet, self).__init__()

    def sample_action(self, action_distribution=None):
        if not action_distribution:
            action_distribution = self.out
        action_distribution = action_distribution.cpu().data.numpy()
        action = np.random.choice(action_distribution, p=action_distribution).flatten()
        action = np.argmax(action_distribution == action)
        return action


class StochasticContinuousNeuralNet(NeuralNet):


    def __init__(self):
        super(StochasticContinuousNeuralNet, self).__init__()


    def sigma(self):
        pass

    def mu(self):
        pass




class SimpleNetwork(NeuralNet):

    def __init__(self, architecture, weight_init=gauss_weights_init(0,0.02),activation_functions=None):
        super(SimpleNetwork, self).__init__()
        if len(architecture) < 2:
            raise Exception("Architecture needs at least two numbers to create network")

        self.activation_functions = activation_functions
        self.layer_list = []
        for i in range(len(architecture)-1):
            self.layer_list.append(nn.Linear(architecture[i], architecture[i+1]))
            setattr(self, "fc" + str(i), self.layer_list[-1])

        self.apply(weight_init)

    def forward(self, x):
        if self.activation_functions:
            for i, func in enumerate(self.activation_functions):
                x = func(self.layer_list[i](x))
        else:
            for i, layer in enumerate(self.layer_list):
                x = self.relu(layer(x))

        i+=1
        while i < len(self.layer_list):
            x = self.layer_list[i](x)
            i+=1

        self.out = x
        return x



class PolicyAHG(StochasticNeuralNet):

    def __init__(self,  input_size, output_size):
        super(PolicyAHG, self).__init__()

        self.f1 = nn.Linear(input_size,32)
        self.f2 = nn.Linear(32, output_size)


    def forward(self, x):

        out = self.f1(x)
        out = self.tanh(out)
        out = self.f2(out)
        out = self.softmax(out)
        self.out = out
        return out



class PolicySPG(StochasticNeuralNet):

    def __init__(self,  input_size, output_size):
        super(PolicySPG, self).__init__()

        self.f1 = nn.Linear(input_size,64)
        self.f2 = nn.Linear(64, output_size)


    def forward(self, x):

        out = self.f1(x)
        out = self.relu(out)
        out = self.f2(out)
        out = self.softmax(out)
        self.out = out
        return out

import nengo
from nengolib.neurons import Tanh

class Reservoir(SpikingNetwork):


    def __init__(self,dt, sim_dt, input_size, network_size=800, recursive=False,
                 spectral_radius=1.,neuron_type=Tanh(),noise=False, synapse_filter=15e-4):


        super(Reservoir, self).__init__(dt, sim_dt)
        self.model = nengo.Network(seed=60)
        self.state = np.zeros(input_size)
        with self.model as model:
            """
                Network configurations.
            """
            self.input_node = nengo.Node(lambda t: self.state)
            # Noise
            if noise:
                noise = nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0,0.5),default_size_out=input_size)

            # If specified create reservoir for the state
            state_ensemble = nengo.Ensemble(network_size, neuron_type=neuron_type, dimensions=input_size, radius=1.2)

            if recursive:
                # l2 = nengo.Ensemble(200, dimensions=observation_size)
                W = np.random.uniform(-0.5, 0.5, (state_ensemble.n_neurons, state_ensemble.n_neurons))
                eig, eigv = np.linalg.eig(W)
                W = W / np.max(np.abs(eig)) * spectral_radius

                print('Spectral radius of reservoir weights: ', spectral_radius)

                nengo.Connection(state_ensemble.neurons, state_ensemble.neurons, transform=W, synapse=synapse_filter)

            """
                Connect input to ensemble.
            """
            nengo.Connection(self.input_node, state_ensemble.neurons,
                             transform=np.random.uniform(-0.5,0.5,(network_size,input_size)), synapse=None)


            """
                This is the output that is going to be read out after simulation ends.
            """
            self.output_probe = nengo.Probe(state_ensemble.neurons)

        self.sim = nengo.Simulator(network=self.model, dt=sim_dt)
        self.dt_steps = int(self.dt / self.sim_dt)

    def forward(self, x, requires_grad=False):
        if tor.is_tensor(x):
            x = x.data.numpy()
        if len(x.shape) >= 2 and x.shape[0] > 1:
            if x.shape[1] != 1 and x.shape[0] != 1:
                return self.batch_forward(x)

        x = x.reshape(-1)
        self.state = x
        self.sim.run_steps(self.dt_steps, progress_bar=False)

        out = self.sim.data[self.output_probe]
        """
            Take the average of all steps as output.
        """
        out = np.mean(out[-self.dt_steps:], axis=0)
        out = out.reshape(1, -1)
        return out


    def batch_forward(self, x):
        out = []
        for i in range(x.shape[0]):
            out.append(self.forward(x[i]))
        return np.vstack(out)

    def reset(self):
        self.sim.close()
        self.sim = nengo.Simulator(network=self.model, dt=self.sim_dt)





class SimpleSpikingAgent(Agent):
    """
        Implements liquid state machine agent with a neural network for readout.
    """

    def __init__(self, spiking_net, readout_net, action_choice_function=return_output):
        super(SimpleSpikingAgent, self).__init__(policy_network=readout_net, action_choice_function=action_choice_function)
        self.spiking_net = spiking_net
        self.spiking_state = None

    def action(self, *args, requires_grad=False):
        if len(args) > 1:
            x = np.hstack(args)
        else:
            x = args[0]
        x = self.spiking_net.forward(x)
        self.spiking_net.reset()
        self.spiking_state = x
        out = self.forward(to_tensor(x))
        return out


    def actions(self, *args, requires_grad=False):
        if len(args) > 1:
            x = np.hstack(args)
        else:
            x = args[0]
        x = self.spiking_net.forward(x)
        self.spiking_net.reset()
        self.spiking_state = x
        out = self.forward(to_tensor(x))
        return out


