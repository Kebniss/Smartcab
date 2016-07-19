import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = ()
        self.q_table = dict()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = ()
        self.next_waypoint = None


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self) # details on traffic light and oncoming traffic
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        State = namedtuple('State', ['light', 'oncoming', 'left', 'right'])
        
        # print "State: ", State
        # print "Inputs: ", inputs 
        tuple_inputs = State(inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
        # print "tuple_inputs: ", tuple_inputs
        self.state = (tuple_inputs, deadline)
        
        # TODO: Select action according to your policy
        action = self.next_waypoint

        # print "\n"
        # print "state: ", self.state
        # print "action: ", action
        # print "\n"

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if not self.q_table:
            self.q_table = dict.fromkeys([self.state, action])
            # print "\n"
            # print "dict:", self.q_table.keys()
            # print "\n"

        # print "if result: ", self.q_table.get((self.state, action)) 
        if self.q_table.get((self.state, action)) is None:
            self.q_table[(self.state, action)] =  reward
            # print "\n"
            # print "first"
            # print "reward:", self.q_table[(self.state, action)]
            # print "\n"
        else:
            self.q_table[(self.state, action)] = self.q_table[(self.state, action)] + reward
            # print "\n"
            # print "other"
            # print "reward:", self.q_table[(self.state, action)]
            # print "\n"

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent - through this the agent gets a random location and heading = 0,1
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
