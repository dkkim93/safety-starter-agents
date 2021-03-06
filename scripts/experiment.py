import gym 
import safe_rl
import os
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from gym_env.wrapper import PendulumCostWrapper
# from utils import set_log


def main(robot, task, algo, seed, exp_name, cpu):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot == 'Doggo':
        num_steps = 1e10
        steps_per_epoch = 60000
    else:
        num_steps = 1e10
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    exp_name = algo
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # if not os.path.exists("./log"):
    #     os.makedirs("./log")
    args.log_name = \
        "seed::" + str(args.seed) + "_algo::" + args.algo + "_task::" + str(args.obstacle_type) + \
        "_cost_lim::" + str(args.cost_lim)
    # custom_log = set_log(args)

    # Algo and Env
    algo = eval('safe_rl.' + algo)

    # env = gym.make("Pendulum-v0")
    # env._max_episode_steps = 64
    # env = PendulumCostWrapper(env)

    import gym_env
    # Setup pointmass
    env = gym.make("pointmass-v0", args=args)
    lam = 0.95
    cost_lam = 0.95
    pi_lr = 0.001

    algo(env_fn=lambda: env,
         ac_kwargs=dict(hidden_sizes=(16, 16),),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=args.cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs,
         prefix=algo,
         lam=lam,
         cost_lam=cost_lam,
         max_ep_len=1000,
         pi_lr=pi_lr,
         args=args
         )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo_lagrangian')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--obstacle-type', type=int)
    parser.add_argument('--cost-lim', type=float)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name == '') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu)
