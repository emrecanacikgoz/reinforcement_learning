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

"""

import random, util
import baseAgents, policies

class TemporalDifferencePredictionAgent(baseAgents.BaseModelFreePredictionAgent):
    """
    Agent that runs the TD(0) algorithm
    """
    def __init__(self, env, discount = 0.9, policy=None, alpha=0.05, alphaScheduler = None):
        """
        env: Environment of the agent
        discount: The discount factor. Should actually be part of the mdp but this implementation makes the agent select it
        policy: The (initial or to be evaluated) policy of the agent
        alpha: The starting learning rate. Some algorithms will not use this
        alphaScheduler: The alpha scheduler that will update the alpha value after each episode. Some algorithms will not use this
        """
        super().__init__(env, discount, policy, alpha=alpha, alphaScheduler=alphaScheduler)
        if policy:
            self.policy = policy
        else:
            self.policy = policies.RandomPolicy(self)
        
    def newEpisode(self):
        super().newEpisode()
        self.numSteps = 0

            
    def run(self):
        """
        Single step of the TD(0) Algorithm
        Returns False if it encounters a terminal state
        """
        currentState = self.env.getCurrentState()
        if self.isTerminal(currentState):
            return False
        action = self.policy(currentState)
        nextState, reward = self.takeAction(action)
        self.values[currentState] += self.alpha*(reward+self.discount*self.values[nextState]-self.values[currentState])
        self.numSteps += 1
        return True

class SarsaAgent(baseAgents.BaseModelFreeControlAgent):
    def __init__(self, env, discount = 0.9, epsilon=0.3, alpha=0.05, alphaScheduler = None, epsilonScheduler = None):

        super().__init__(env, discount, alpha, alphaScheduler, epsilon, epsilonScheduler)

        """
        You can add your own fields
        """
        
        #You need to pick the correct policy! Set the below field to the correct one
        self.policy = None
    
    """
    You can add your own functions
    """
    
    def newEpisode(self):
        super().newEpisode()
        self.numSteps = 0
        self.actionToTake = self.policy(self.env.getCurrentState())

    def run(self):
        util.raiseNotDefined()
        """
        Single step of the Sarsa Algorithm
        Comment out the previous line and write your code here.
        You need to implement sarsa for a single step
        Do not forget to update self.qvalues
        WARNING: Return False if you encounter a terminal state. This is not for the nextState!
        If the nextState is the terminal state, remember what to do with the Q target
        Feel free to add new methods
        
        return False when the episode ends, return True otherwise!
        """
        
        #return True
        
class QLearningAgent(baseAgents.BaseModelFreeControlAgent):
    def __init__(self, env, discount = 0.9, epsilon=0.3, alpha=0.05, alphaScheduler = None, epsilonScheduler = None):
        super().__init__(env, discount, alpha, alphaScheduler, epsilon, epsilonScheduler)

        """
        You can add your own fields
        """
        
        #You need to pick the correct policy! Set the below field to the correct one
        self.policy = None
        
    """
    You can add your own functions
    """
        
    def newEpisode(self):
        super().newEpisode()
        self.numSteps = 0

    def run(self):
        util.raiseNotDefined()
        """
        Single step of the Q-Learning Algorithm
        Comment out the previous line and write your code here.
        You need to implement q-learning for a single step
        Do not forget to update self.qvalues
        WARNING: Return False if you encounter a terminal state. This is not for the nextState!
        If the nextState is the terminal state, remember what to do with the Q target
        Feel free to add new methods
        
        return False when the episode ends, return True otherwise!
        """
        
        #return True
       