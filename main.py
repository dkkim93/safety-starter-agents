import gym
import argparse
from gym_env.wrapper import PendulumCostWrapper
from safe_rl import ppo_lagrangian


def main(args):
    env = gym.make(args.env_name)
    env._max_episode_steps = args.ep_horizon
    if args.env_name == "Pendulum-v0":
        env = PendulumCostWrapper(env)
    
    ppo_lagrangian(
        env_fn=lambda: env,
        ac_kwargs=dict(hidden_sizes=(64, 64)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="safe-rl")

    # Env
    parser.add_argument(
        "--env-name", type=str, default="Pendulum-v0",
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-horizon", type=int, default=150,
        help="Episode terminates when episodic horizon is reached")

    # Misc
    parser.add_argument(
        "--seed", type=int, default=1, 
        help="Seed for env, numpy and torch")
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix for logging")

    args = parser.parse_args()

    # Set log name
    args.log_name = "env::%s_seed::%s_prefix::%s_log" % (args.env_name, args.seed, args.prefix)

    # Begin main code
    main(args=args)
