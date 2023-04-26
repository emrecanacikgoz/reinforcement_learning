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

"""
This class is given to you as an example
"""
class MonteCarloPredictionAgent(baseAgents.BaseModelFreePredictionAgent):
    """
    Agent that runs the MC prediction algorithm
    """
    def __init__(self, env, discount = 0.9, policy = None):
        """
        env: Environment of the agent
        discount: The discount factor. Should actually be part of the mdp but this implementation makes the agent select it
        policy: The (initial or to be evaluated) policy of the agent
        """
        super().__init__(env, discount, policy)
        self.visitCount = util.Counter()
        self.numEpisodes = 0
        
        if policy:
            self.policy = policy
        else:
            self.policy = policies.RandomPolicy(self)
        
    def _doOneEpisode(self):
        """
        This runs a single episode on the environment with the given policy
        """
        transitionList = []
        currentState = self.env.getCurrentState()
        while True:
            action = self.policy(currentState)
            nextState, reward = self.takeAction(action)
            transitionList.append((currentState,action,reward))
            if self.isTerminal(nextState):
                break
            currentState = nextState
        return transitionList
    
    def run(self):
        """
        Every-visit MC prediction algorithm for one episode.
        Doesn't need to return anything special
        """
        
        self.newEpisode()

        # Run one episode and get transitions
        # Transitions are a list of state,action,reward tuples
        transitionList = self._doOneEpisode()

        # Reverse due to the backward recursive definition, do not want to keep extra memory:
        # G(i-1) = R(i-1) + gamma*G(i), G(end) = R(end)
        transitionList.reverse()

        currentReturn = 0
        for transition in transitionList:
            state, action, reward = transition
            # Increment the visit count
            self.visitCount.increment(state)
            # Update the return
            currentReturn = reward + self.discount*currentReturn
            # Update the value
            self.values[state] +=  1./self.visitCount[state]*(currentReturn - self.values[state])

        return self.values

class MonteCarloControlAgent(baseAgents.BaseModelFreeControlAgent):
    def __init__(self, env, discount = 0.9,  epsilon = 0.3, policy = None, epsilonScheduler = None):
        super().__init__(env, discount, epsilon, epsilonScheduler = epsilonScheduler)
        """
        env: Environment of the agent
        discount: The discount factor. Should actually be part of the mdp but this implementation makes the agent select it
        policy: The (initial or to be evaluated) policy of the agent
        epsilon: The initial epsilon value for epsilon greedy action selection
        epsilonScheduler: The epsilon scheduler that will update the epsilon value after each episode. 
        """
        
        self.visitCount = util.Counter()
        """
        You can add your own fields. 
        """
        
        #You need to pick the correct policy! Set the below field to the correct one
        self.policy = None
        
    """
    You can add your own functions
    """
        

    def run(self):
        self.newEpisode()
        util.raiseNotDefined()
        """
        Comment out the previous line and write your code here.
        You need to implement every-visit MC control algorithm for one episode
        After this method is run, self.qvalues must be filled with the updated q-values
        Make sure to look at policies.PolicyFromQValues and the BaseModelFreeValueControlAlgo
        Feel free to add new methods
        Feel free to look at the prediction version
        Note that you need to implement the do one episode that is using your chosen policy
        Doesn't need to return anything special
        
        return value is not important
        """ 

        #return self.qvalues
