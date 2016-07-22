import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
from pandas import value_counts
from math import log

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = ()
        self.q_table = dict()
        self.k = 0 # to be used in e greedy implementation
        self.outcomes = [] # to calculate overall success rate
        self.start = True


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.k += 1
        self.state = ()
        self.next_waypoint = None
        self.outcomes += [False]


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self) # details on traffic light and oncoming traffic
        deadline = self.env.get_deadline(self)

        # expiring = False
        # if deadline <= 5:
        #     expiring = True

        # TODO: Update state
        self.state = get_state(inputs)
        
        # TODO: Select action according to your policy
        action = get_action(self.k, self.next_waypoint, self.q_table, self.state)

        # print "\n"
        # print "state: ", self.state
        # print "action: ", action
        # print "\n"

        # Execute action and get reward
        reward = self.env.act(self, action)

        next_inputs = self.env.sense(self)
        next_state = get_state(next_inputs)

        # TODO: Learn policy based on state, action, reward
        if not self.q_table:
            # create the dict
            self.q_table = dict.fromkeys([self.state, action])
            # print "\n"
            # print "dict:", self.q_table.keys()
            # print "\n"

        alpha = 1/log(self.k+2) 
        gamma = 0.5

        if self.q_table.get((self.state, action)) is None:
            self.q_table[(self.state, action)] = 0

        max_q = None
        next_action = 'noAction'
        for act in ['None', 'left', 'forward', 'right']:
            q_value_s_a = self.q_table.get((next_state, act))
            if q_value_s_a is not None and q_value_s_a > max_q:
                max_q = q_value_s_a
                next_action = act
        
        if next_action == 'noAction':
            self.q_table[(next_state, next_action)] = 0

        self.q_table[(self.state, action)] = (1 - alpha) * self.q_table.get((self.state, action)) + alpha * (
                                            reward + gamma * self.q_table.get((next_state, next_action)))

       
        # check if the agent reached the destination
        location = self.env.agent_states[self]["location"]
        destination = self.env.agent_states[self]["destination"]
        if location == destination:
            self.outcomes[-1] = True

        # when the end of the simulation is reached print overall success rate
        end = 100 # number of trials defined in run
        if self.k == end:
            successes = value_counts(self.outcomes)[True]
            succ_rate = (successes * 1.0) / len(self.outcomes)
            print "\nSuccess rate: {}".format(succ_rate)
            print "\n"

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def get_state(inputs):
    """ Transforms inputs from dict to named tuple"""
    State = namedtuple('State', ['light', 'oncoming', 'left', 'right'])
    tuple_inputs = State(inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
    return(tuple_inputs)


def get_action(k, next_waypoint, q_table, state):
    """ Applies epsilon-greedy algorithm and returns the action to take. """

    
    actions = [None, next_waypoint] #'forward', 'left', 'right'
    # action = actions[random.randint(0,3)]
    epsilon = 1/log(k+2) 
    rnd = random.uniform(0,1)

    if rnd <= epsilon:
        action = actions[random.randint(0,1)]
    else: 
        max_q = None
        action = next_waypoint
        for act in ['None', 'forward', 'left', 'right']:
            q_value_s_a = q_table.get((state, act))
            if q_value_s_a is not None and q_value_s_a > max_q:
                max_q = q_value_s_a
                action = act

    return action

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent - through this the agent gets a random location and heading = 0,1
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
