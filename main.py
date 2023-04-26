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

DO NOT MODIFY

"""

import random
import sys
import mdp
import environment
import util
import optparse
import gridworld
from time import sleep
import policies

def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in epsilon-greedy policies (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=100,
                         metavar="I", help='Maximum number of iterations for DP methods (default %default).')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes to run for RL methods (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="BookGrid",
                         help='Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, MazeGrid, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=200,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--algorithm',action='store', metavar="A",
                         type='string',dest='algo',default="mcp",
                         help='Agent to run (options are pe: Policy Evaluation, qi: Q-Value Iteration (Not Q-Learning!), pe:Policy Iteration, mcp: Monte Carlo Prediction, mcc: Monte Carlo Control, td: Temporal Difference Prediction, sr: Sarsa, ql: Q-Learning, default: %default)')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent')
    #optParser.add_option('-v', '--valueSteps',action='store_true', dest='valueSteps', default=False,
    #                     help='Display each step of value iteration')
    optParser.add_option('-x', '--displayQValues',action='store_true',
                         dest='dQv',default=False,
                         help='Whether to q-values or values')

    opts, args = optParser.parse_args()

    if opts.manual:
        print('## Disabling Agents in Manual Mode (-m) ##')
        opts.agent = None

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
    # if opts.quiet:
        opts.pause = False
        # opts.manual = False

    if opts.manual:
        opts.pause = True

    return opts

if __name__ == '__main__':

    random.seed(123456)

    opts = parseOptions()

    ###########################
    # GET THE GRIDWORLD
    ###########################

    mdpFunction = getattr(gridworld, "get"+opts.grid)
    mdp = mdpFunction()
    mdp.setLivingReward(opts.livingReward)
    mdp.setNoise(opts.noise)
    mdp.setDiscount(opts.discount)

    ###########################
    # GET THE DISPLAY ADAPTER
    ###########################

    import textGridworldDisplay
    display = textGridworldDisplay.TextGridworldDisplay(mdp)
    if not opts.textDisplay:
        import graphicsGridworldDisplay
        display = graphicsGridworldDisplay.GraphicsGridworldDisplay(mdp, opts.gridSize, opts.speed)
    try:
        display.start()
    except KeyboardInterrupt:
        sys.exit(0)

    values = util.Counter()
    qvalues = util.Counter()
    policy = None

    # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES
    #try:
    #    if not opts.manual:
    #        if opts.valueSteps:
    #            for i in range(opts.iters):
    #                tempAgent = valueIterationAgents.ValueIterationAgent(mdp, opts.discount, i)
    #                display.displayValues(tempAgent, message = "VALUES AFTER "+str(i)+" ITERATIONS")
    #                display.pause()
    #
    #        display.displayValues(values, message = "VALUES AFTER "+str(opts.iters)+" ITERATIONS")
    #        display.pause()
    #        #display.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.iters)+" ITERATIONS")
    #        #display.pause()
    #except KeyboardInterrupt:
    #    sys.exit(0)

    # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)

    if opts.algo == 'qi' or opts.algo == 'sr' or opts.algo == 'ql' or opts.algo == 'srl' or opts.algo == 'mcc':
        opts.dQv = True
        
    if opts.algo == 'pe' or opts.algo == 'qi' or opts.algo == 'pi' or opts.algo == 'vi':
        vizIter = True
    else:
        vizIter = False

    displayCallback = lambda x: None
    if not opts.quiet:
        if opts.manual and opts.agent == None:
            displayCallback = lambda state: display.displayNullValues(state)
        else:
            if opts.dQv:
                displayCallback = lambda state: display.displayQValues(qvalues, currentState=state)
            else:
                displayCallback = lambda state: display.displayValues(values, currentState=state)

    #Baris: Mainly ignoring this
    messageCallback = lambda x: gridworld.printString(x)
    if opts.quiet:
        messageCallback = lambda x: None

    # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
    pauseCallback = lambda : None
    if opts.pause:
        pauseCallback = lambda : display.pause()


    ###########################
    # MODIFIED ENV TAKING DISPLAY AND PAUSE
    ###########################
    env = gridworld.GridworldEnvironment(mdp, displayCallback, pauseCallback)

    # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)
    if opts.manual:
        # RUN EPISODES
        if opts.episodes > 0:
            print
            print("RUNNING", opts.episodes, "EPISODES")
            print
        returns = 0
        for episode in range(1, opts.episodes+1):
            returns += gridworld.runManual(env,mdp.discount,messageCallback,episode)
        if opts.episodes > 0:
            print
            print ("AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes))
            print
            print
        sys.exit(0)


    # RUN THE ALGORITHMS
    import dpAgents, mcAgents, tdAgents

    #policy = None

    # Policy evaluation, implemented for you
    if opts.algo == 'pe':
        pe = dpAgents.PolicyEvaluationAgent(mdp, env, discount = opts.discount, maxIters = opts.iters)
        opts.iters = pe.run()
        values = pe.getValues()
    
    # Q-Value Iteration, you need to implement this
    elif opts.algo == 'qi':
        qi = dpAgents.QValueIterationAgent(mdp, env, discount = opts.discount, maxIters = opts.iters)
        opts.iters=qi.run()
        qvalues = qi.getQValues()
        policy = qi.policy
    
    # Policy Iteration, you need to implement this
    elif opts.algo == 'pi':
        pi = dpAgents.PolicyIterationAgent(mdp, env, discount = opts.discount)
        opts.iters = pi.run()
        values = pi.getValues()
        policy = pi.policy
		
        #For debugging:
        #pe = dpAgents.PolicyEvaluationAgent(mdp, env, discount = opts.discount, maxIters = 100, policy=policy)
        #pe.run()
        #dbg_values = pe.getValues()
    
    # Monte Carlo Prediction, implemented for you
    elif opts.algo == 'mcp':
        mcp = mcAgents.MonteCarloPredictionAgent(env, discount = opts.discount)
        mcp.getPossibleActions = mdp.getPossibleActions
        mcp.isTerminal = mdp.isTerminal
        for i in range(0, opts.episodes):
            env.reset()
            mcp.run()
            values = mcp.getValues()
            # display.displayValues(values)
            # display.pause()

    # Monte Carlo Control, you need to implement this
    elif opts.algo == 'mcc':
        mcc = mcAgents.MonteCarloControlAgent(env, discount = opts.discount)
        mcc.getPossibleActions = mdp.getPossibleActions
        mcc.isTerminal = mdp.isTerminal
        for i in range(0, opts.episodes):
            env.reset()
            mcc.run()
            qvalues = mcc.getQValues()
            #display.displayQValues(qvalues)
            #display.pause()
            
    # Temporal Difference Prediction, you need to implement this
    elif opts.algo == 'td':
        td = tdAgents.TemporalDifferencePredictionAgent(env, discount = opts.discount)
        td.getPossibleActions = mdp.getPossibleActions
        td.isTerminal = mdp.isTerminal
        for i in range(0, opts.episodes):
            env.reset()
            td.newEpisode()
            while(td.run()): #run until the end of an episode
                values=td.values
                #display.displayValues()
                #display.pause()
    
    # Sarsa control, you need to implement this
    elif opts.algo == 'sr': 
        sr = tdAgents.SarsaAgent(env, discount = opts.discount, epsilon = opts.epsilon)
        sr.getPossibleActions = mdp.getPossibleActions
        sr.isTerminal = mdp.isTerminal
        for i in range(0, opts.episodes):
            env.reset()
            sr.newEpisode()
            while(sr.run()): #run until the end of an episode
                qvalues=sr.getQValues()
                #display.displayQValues(qvalues, sr.getPolicy())
                #display.pause()
        policy=sr.getPolicy()

    # Q-Learning, you need to implement this
    elif opts.algo == 'ql':
        ql = tdAgents.QLearningAgent(env, discount = opts.discount, epsilon = opts.epsilon)
        ql.getPossibleActions = mdp.getPossibleActions
        ql.isTerminal = mdp.isTerminal
        for i in range(0, opts.episodes):
            env.reset()
            ql.newEpisode()
            while(ql.run()): #run until the end of an episode
                qvalues=ql.getQValues()
                #display.displayQValues(qvalues, ql.getPolicy())
                #display.pause()
        policy=ql.getPolicy()

    #for key in values.keys():
    #    print(key,values[key])
    #print(qvalues)

    # DISPLAY POST-LEARNING VALUES / Q-VALUES
    if not opts.manual:
        try:
            if opts.dQv and len(qvalues) != 0:
                if vizIter:
                    display.displayQValues(qvalues, message = "Q-VALUES AFTER "+str(opts.iters)+" ITERATIONS")
                else:
                    display.displayQValues(qvalues, message = "Q-VALUES AFTER "+str(opts.episodes)+" EPISODES")
                display.pause()
                sleep(1.0)
            else:
                if vizIter:
                    display.displayValues(values, policy=policy, message = "VALUES AFTER "+str(opts.iters)+" ITERATIONS")
                else:
                    display.displayValues(values, policy=policy, message = "VALUES AFTER "+str(opts.episodes)+" EPISODES")
                display.pause()
                sleep(1.0)
        except KeyboardInterrupt:
            sys.exit(0)
    import graphicsUtils
        
    #graphicsUtils.writePostscript('latest'+opts.algo+'.eps')
    graphicsUtils.writePng('latest_'+opts.algo+'_'+opts.grid+'.png')

