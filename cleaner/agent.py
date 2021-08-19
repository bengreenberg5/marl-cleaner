from cleaner.utils import RAY_DIR


class Agent(object):
    def __init__(self, policy_name):
        assert policy_name in ["ppo", "dqn"], f"unknown policy name: {policy_name}"
        self.policy_name = policy_name
        self.trainer = None

    def load_from_checkpoint(self, run_name, checkpoint_num):
        checkpoint_path = f"{self.results_dir}/checkpoint_{str(checkpoint_num).zfill(6)}/checkpoint-{checkpoint_num}"
        self.trainer.restore(checkpoint_path)

    def prepare_to_run(self, run_name, agent_num, checkpoint_num=None):
        """
        Note: if loading a checkpoint, env dimensions must match.
        Otherwise, the Trainer won't be able to interpret input from observations.
        """
        self.results_dir = f"{RAY_DIR}/{run_name}"
        self.run_name = run_name
        self.agent_num = agent_num
        self.name = f"{run_name}:{agent_num}"
        if checkpoint_num:
            self.load_from_checkpoint(run_name, checkpoint_num)
