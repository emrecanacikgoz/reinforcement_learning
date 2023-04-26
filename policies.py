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

import random, util, math

class BasePolicy():
    """
    The base policy class
    """
    def __init__(self, agent, epsilon = 0, epsilonGreedy =  False, returnProbabilities = False):
        """
        agent: The agent that is using the policy
        epsilon: The default epsilon to use
        epsilonGreedy: To specify whether the behavior is epsilon-greedy by default
        returnProbabilities: Whether to return action distribution instead of the action by default
        """
        self.agent = agent
        self.epsilon = epsilon
        self.epsilonGreedy = epsilonGreedy and (self.epsilon != 0)
        self.returnProbabilities = returnProbabilities and (not self.epsilonGreedy)


    def greedyAction(self, state):
        """
        Returns the greedy action for the given policy, breaking ties randomly
        i.e. argmax_a( Q(s,a) ) or argmax_a( sum_s'( R(s,a,s')+discount*P(s'|s,a)*V(s') ) )
        """
        pass

    def policyProbs(self, state):
        """
        Returns a dictionary or util.Counter() representing pi(a|s), with states as keys and actions as values
        For a deterministic policy, only one state will have 1.0, the rest will be 0.0.
        For a random policy, each will be 1/m where m is the number of actions
        """
        pass

    def epsilonGreedyAction(self, state, epsilon = -1):
        #Do not change the if statement
        if(epsilon < 0):
            epsilon = self.agent.epsilon

        """
        Comment out the previous line and write your code here.
        You need to implement epsilon-greedy action selection from a deterministic policy here.
        This method should return an action obtained from the MDP.
        Hints:
        - You can use the greedyAction function of this class
        - Look at the self.agent.getPossibleActions function for all possible actions
        """

        actions = self.mdp.getPossibleActions(state)
        if random.random() < epsilon:
            return random.choice(actions)
        else:
            return self.greedyAction(state)
        
    def policy(self,state):
        """
        This isn't really explicitly needed as you should know which policy to call given an algorithm
        """
        if(self.returnProbabilities):
            return self.policyProbs(state)
        else:
            if(self.epsilonGreedy):
                return self.epsilonGreedyAction(state)
            else:
                return self.greedyAction(state)
                
    def __call__(self, state):
        """
        Allows us to call policy(state)
        """
        return self.policy(state)
        
    def __getitem__(self, state):
        """
        Allows us to call policy[state]
        """
        return self.policy(state)
        
    def __contains__(self, key):
        """
        Allows us to call state in policy
        Only works when the agent has an mdp field!
        """
        return key in self.agent.mdp.getStates() 

class RandomPolicy(BasePolicy):
    """
    Purely random policy. 
    """
    def __init__(self, agent, returnProbabilities = False):
        super().__init__(agent, returnProbabilities = returnProbabilities)

    def greedyAction(self, state):
        actions = self.agent.getPossibleActions(state)
        return random.choice(actions)

    def policyProbs(self,state):
        actions = self.agent.getPossibleActions(state)
        probs = util.Counter() #can normalize if needed
        for action in actions:
            probs[action] = 1/len(actions)
        return probs

class SingleActionPolicy(BasePolicy):
    """
    Policy that always selects the given action or the first action if the given action is not legal
    """
    def __init__(self, agent, action, returnProbabilities = False):
        super().__init__(agent, returnProbabilities = returnProbabilities)
        self.action = action

    def greedyAction(self, state):
        actions = self.agent.getPossibleActions(state)
        if self.action in actions:
            return self.action
        else:
            return actions[0]

    def policyProbs(self,state):
        actions = self.agent.getPossibleActions(state)
        probs = util.Counter() #can normalize if needed
        if self.action in actions:
            probs[self.action] = 1.0
        else:
            probs[actions[0]] = 1.0
        return probs

class TabularPolicy(BasePolicy):
    """
    Representation of the tabular policy. 
    Explicitly holds pi(state) = action with a dictionary like object (the policyTable)
    """
    def __init__(self, agent, policyTable = None):
        super().__init__(agent)
        if not policyTable:
            policyTable = {}
        self.policyTable = policyTable
        for state in list(self.agent.values.keys()):
            if state not in self.policyTable.keys():
                if self.agent.isTerminal(state):
                    self.policyTable[state] = None
                    continue
                actions = self.agent.getPossibleActions(state)
                self.policyTable[state] = random.choice(actions)

        
    def greedyAction(self, state):
        return self.policyTable[state]

    def policyProbs(self,state):
        probs = util.Counter()
        probs[self.policyTable[state]] = 1.0
        return probs
        
    def __setitem__(self,state,action):
        self.policyTable[state] = action

class PolicyFromQValues(BasePolicy):
    """
    The policy object that calculates actions from the agent Q-Values.
    Agent must be a control agent but we are not checking for that
    
    You need to complete this yourself!
    """
    def __init__(self, agent):
        if hasattr(agent,'epsilon'):
            super().__init__(agent, epsilon=agent.epsilon)
        else:
            super().__init__(agent)
        self.agent = agent


    def greedyAction(self, state):
        """
        Return the greedy action for the given state using agent.getActionValuesGivenState(state)
        """
        if self.agent.isTerminal(state):
            return None
        qVals = self.agent.getActionValuesGivenState(state)
        maxVal = -math.inf
        argMaxAct = None
        for action, value in qVals.items():
            if maxVal < value:
                maxVal = value
                argMaxAct = action
        return argMaxAct
        
        
    def policyProbs(self,state):
        """
        This is a bit ambigious but let's return the "greedy version"
        """
        probs = util.Counter()
        probs[self.greedyAction(state)] = 1.0
        return probs

       