class Trainer:

    def __init__(self, env, agent1, agent2):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2

    def run_episode(self):

        obs,_ = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            a1 = self.agent1.act(obs["mover1"])
            a2 = self.agent2.act(obs["mover2"])

            obs, rewards, done, _, _ = self.env.step({
                "mover1": a1,
                "mover2": a2
            })

            total_reward += sum(rewards.values())

        return total_reward