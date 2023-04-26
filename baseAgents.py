"""
Copyright 2019 Baris Akgun

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided
with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Most importantly, this software is provided for educational purposes and should not be used for
anything else.

AUTHORS: Baris Akgun

Specific Notes:
The source code provide in this file is not designed with modularity, maintainability or any
other software engineering principles in mind. It was mainly designed to eliminate code duplication
in other files and to keep everything in one single place. As such there are portions which are not
useful or even meaningful for developing certain algorithms (e.g. getting the greedy actions for
prediction algorithms).

DO NOT MODIFY

"""

import util, random
import parameterSchedulers

class BaseValueAgent():
    """
    Base class for the agents
    """   
    def __init__(self, env, discount = 0.9, policy = None):
        """
        env: Environment of the agent
        discount: The discount factor. Should actually be part of the mdp but this implementation makes the agent select it
        policy: The (initial or to be evaluated) policy of the agent
        """
        
        self.discount = discount
        self.env = env
        self.policy = policy

        self.values = util.Counter()
        self.qvalues = util.Counter()
        self.numEpisodes = 0

    def run(self):
        """
        The function that calls the agent's algorithm. 
        Not the best abstraction since different algorithms do things differently;
        (1) running offline, (2) running a single episode or (3) running a single step
        """
        pass

    def newEpisode(self):
        """
        Any initializations before a new episode. Some algorithms need it, some don't.
        """
        self.numEpisodes += 1
	
    def getValue(self, state):
        """
        Return V(state)
        """
        return self.values[state]

    def getValues(self):
        """
        Return a dictionary or a util.Counter() with keys as states and values as V(state)
        """
        return self.values

    def getQValue(self, state, action):
        """
        Return Q(state,action)
        """
        return self.qvalues[(state,action)]

    def getQValues(self):
        """
        Return a dictionary or a util.Counter() with keys as (state,action) pairs and values as Q(state,action)
        i.e. Q[(state,action)] must map to the desired q-value
        """
        return self.qvalues

    def getActionValuesGivenState(self, state):
        """
        Return a list of values corresponding to Q(s,.) for each action
        """
        actions = self.env.getPossibleActions(state)
        qVals = {}
        for action in actions:
            qVals[action] = self.qvalues[(state,action)]
        return qVals
        
    def getPolicy(self):
        """
        Returns the current policy of the agent
        """
        return self.policy
        
    def getGreedyAction(self,state):
        """
        Returns the greedy action in the given state. 
        """
        return self.policy.greedyAction(state)
        
    def getEpsilonGreedyAction(self,state):
        """
        Returns the action selected from the soft policy (e.g. epsilon-greedily) in the given state. 
        """
        return self.policy.epsilonGreedyAction(state)
        
    def getActionDistribution(self,state):
        """
        Returns the action distributions given the state as a dictionary or a Counter 
        """
        return self.policy.policyProbs(state)
        
    def takeAction(self, action):
        """
        Takes an action in the environment
        Returns the next state and the reward
        """
        return self.env.doAction(action)
        
    def getPossibleActions(self, state):
        """
        Returns the possible actions given the state.
        """
        pass
        
    def isTerminal(self, state):
        """
        Returns true if the agent is in the terminal state, false otherwise
        """
        pass

class BaseDpAgent(BaseValueAgent):
    """
    Base class for the agents that will use dynamic programming
    """
    
    def __init__(self, mdp, env, discount = 0.9, policy = None):
        """
        mdp: The underlying Markov Decision Process 
        env: Environment of the agent
        discount: The discount factor. Should actually be part of the mdp but this implementation makes the agent select it
        """
        super().__init__(env, discount, policy)  
        self.mdp = mdp
        
    def getPossibleActions(self, state):
        return self.mdp.getPossibleActions(state)
    
    def isTerminal(self,state):
        return self.mdp.isTerminal(state)

class BaseModelFreePredictionAgent(BaseValueAgent):
    """
    Base class for the model free pediction agent
    """
    def __init__(self, env, discount = 0.9, policy = None, alpha = 0.01, alphaScheduler = None):
        """
        env: Environment of the agent
        discount: The discount factor. Should actually be part of the mdp but this implementation makes the agent select it
        policy: The (initial or to be evaluated) policy of the agent
        alpha: The starting learning rate. Some algorithms will not use this
        alphaScheduler: The alpha scheduler that will update the alpha value after each episode. Some algorithms will not use this
        """
        super().__init__(env, discount, policy)
        self.alpha = alpha
        if not alphaScheduler:
            alphaScheduler = parameterSchedulers.NoneScheduler(alpha)
        self.alphaScheduler = alphaScheduler
        
    def newEpisode(self):
        super().newEpisode()
        self.alpha = self.alphaScheduler.update()

class BaseModelFreeControlAgent(BaseModelFreePredictionAgent):
    def __init__(self, env, discount = 0.9, alpha = 0.01, alphaScheduler = None, epsilon = 0.3, epsilonScheduler = None):
        """
        env: Environment of the agent
        discount: The discount factor. Should actually be part of the mdp but this implementation makes the agent select it
        alpha: The starting learning rate. Some algorithms will not use this
        alphaScheduler: The alpha scheduler that will update the alpha value after each episode. Some algorithms will not use this
        epsilon: The initial epsilon value for epsilon greedy action selection
        epsilonScheduler: The epsilon scheduler that will update the alpha value after each episode. Some algorithms will not use this
        """
        super().__init__(env, discount, alpha=alpha, alphaScheduler=alphaScheduler)
        
        self.epsilon = epsilon
        if not epsilonScheduler:
            epsilonScheduler = parameterSchedulers.NoneScheduler(epsilon)
        self.epsilonScheduler = epsilonScheduler
        
    def newEpisode(self):
        super().newEpisode()
        self.epsilon = self.epsilonScheduler.update()


