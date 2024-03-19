#!/usr/bin/env python3
"""
Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
"""
import concurrent.futures
import multiprocessing
import sys
import timeit

import numpy as np
import pickle
import gymnasium as gym

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False
parallelism = 5

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid


def initialize_model():
    model = {}
    model["W1"] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model["W2"] = np.random.randn(H) / np.sqrt(H)
    episode_number = 0

    return episode_number, model


def save_checkpoint(fname, model, episode_num):
    checkpoint = {"model": model, "episode_number": episode_num}
    pickle.dump(checkpoint, open(fname, "wb"))


def load_checkpoint(fname):
    checkpoint = pickle.load(open(fname, "rb"))

    if "episode_number" in checkpoint:
        episode_number = checkpoint.get("episode_number")
        model = checkpoint.get("model")
    else:
        episode_number = 0
        model = checkpoint

    return episode_number, model


if resume:
    episode_number, model = load_checkpoint("save.p")
else:
    episode_number, model = initialize_model()

grad_buffer = {
    k: np.zeros_like(v) for k, v in model.items()
}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    """
    The sigmoid function is good for binary classification, because you can interpret the output as
    the probability to pick choice 1, with 1-p being the probability to pick choice 2.

    When there are multiple categories, a softmax activation function can provide a distribution over the
    categories where the probabilities sum to 1.
    """
    return 1.0 / (
        1.0 + np.exp(-x)
    )  # sigmoid "squashing" function to interval [0,1], probability of "up"


def prepro(I):
    """prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.cfloat).ravel()


# def discount_rewards(r):
#     """take 1D float array of rewards and compute discounted reward"""
#     discounted_r = np.zeros_like(r)
#     running_add = 0
#     for t in reversed(range(0, r.size)):
#         if r[t] != 0:
#             running_add = (
#                 0  # reset the sum, since this was a game boundary (pong specific!)
#             )
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#     return discounted_r


def discounted_rewards(rewards):
    """
    Takes in a 1D float array of rewards (rewards) and computes a time-discounted reward.
    """
    rewards_discounted = np.zeros_like(r)

    running_reward = 0

    for step in reversed(range(rewards.size)):
        if rewards[step] != 0:
            # In Pong, rewards only appear at a game boundary, so we reset the reward
            running_reward = 0

        running_reward = (
            running_reward * gamma + rewards[step]
        )  # Discount by gamma for each step away from the reward
        rewards_discounted[step] = running_reward

    return rewards_discounted


def policy_forward(x):
    h = np.dot(model["W1"], x)
    h[h < 0] = 0  # ReLU nonlinearity activation function
    logp = np.dot(model["W2"], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(epx, eph, epdlogp):
    """backward pass. (eph is array of intermediate hidden states)"""
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model["W2"])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {"W1": dW1, "W2": dW2}


def episode_worker(thread_id):
    """
    Executes a single episode.
    """
    start = timeit.default_timer()

    env = gym.make("ALE/Pong-v5")

    observation, game_state = env.reset()  # Get the first game state
    prev_x = None  # used in computing the difference frame

    xs, hs, dlogps, drs = [], [], [], []
    reward_sum = 0

    while True:
        # Preprocess the frame
        cur_x = prepro(observation)

        # Calculate the difference frame (motion)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # Run the forward pass of the policy network and sample an action
        aprob, h = policy_forward(x)
        action = (
            2 if np.random.uniform() < aprob else 3
        )  # In Pong, 3 is LEFT and 2 is RIGHT

        # Store various intermediate values used in backpropagation
        # TODO: Understand these variables
        xs.append(x)  # Observation
        hs.append(h)  # hidden state
        y = 1 if action == 2 else 0  # A "fake label"
        dlogps.append(
            y - aprob
        )  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/losses if confused)

        # step the environment and get new measurements
        # Truncated is true if the game ends early
        observation, reward, done, _truncated, info = env.step(action)
        reward_sum += reward

        drs.append(
            reward
        )  # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            print(f"{thread_id} is DONE.")
            break

    # Clean up the game simulation
    env.close()

    end = timeit.default_timer()

    print(f"{thread_id} time taken: {end-start}s")

    # Stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)

    return epx, eph, epdlogp, epr, reward_sum, thread_id


def episode_worker_mp(thread_id, result_queue):
    """
    Executes a single episode using multiprocessing.
    """
    start = timeit.default_timer()

    env = gym.make("ALE/Pong-v5")

    observation, game_state = env.reset()  # Get the first game state
    prev_x = None  # used in computing the difference frame

    xs, hs, dlogps, drs = [], [], [], []
    reward_sum = 0

    while True:
        # Preprocess the frame
        cur_x = prepro(observation)

        # Calculate the difference frame (motion)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # Run the forward pass of the policy network and sample an action
        aprob, h = policy_forward(x)
        action = (
            2 if np.random.uniform() < aprob else 3
        )  # In Pong, 3 is LEFT and 2 is RIGHT

        # Store various intermediate values used in backpropagation
        # TODO: Understand these variables
        xs.append(x)  # Observation
        hs.append(h)  # hidden state
        y = 1 if action == 2 else 0  # A "fake label"
        dlogps.append(
            y - aprob
        )  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/losses if confused)

        # step the environment and get new measurements
        # Truncated is true if the game ends early
        observation, reward, done, _truncated, info = env.step(action)
        reward_sum += reward

        drs.append(
            reward
        )  # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            print(f"{thread_id} is DONE.")
            break

    # Clean up the game simulation
    env.close()

    end = timeit.default_timer()

    print(f"{thread_id} time taken: {end-start}s")

    # Stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)

    result_queue.put((epx, eph, epdlogp, epr, reward_sum, thread_id))


def simulate_batch(episode_num, batch_size):
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallelism) as executor:
        try:
            # Submit a batch of jobs
            futures = [
                executor.submit(episode_worker, f"EP#{episode_num}-WORKER#{i}")
                for i in range(batch_size)
            ]

            # Wait for all processes to complete
            concurrent.futures.wait(futures)

            # Collect results
            results = [future.result() for future in futures]
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating child processes...")
            executor.shutdown(
                wait=True, cancel_futures=True
            )  # Initiate the shutdown process
            sys.exit(1)  # Exit the main process

    return results


def simulate_batch_mp(episode_num, batch_size):
    mp_context = multiprocessing.get_context("fork")
    manager = mp_context.Manager()
    results = []
    result_queue = manager.Queue()
    process_ct = 0

    while process_ct <= batch_size:
        active_processes = []

        for _ in range(parallelism):
            p = mp_context.Process(
                target=episode_worker_mp,
                args=(f"EP#{episode_num}-WORKER#{process_ct}", result_queue),
                daemon=True,
            )
            p.start()
            active_processes.append(p)

            process_ct += 1

            # Don't exceed batch_size
            if process_ct > batch_size:
                break

        for p in active_processes:
            p.join()

        while not result_queue.empty():
            results.append(result_queue.get())

        print("Results captured")

    return results


def reshape_and_concat(storage_arr, batch_of_samples):
    """
    If you have a storage array of size Bx1xN and a sample of size BxN, this function will concatenate
    the samples onto the Bx1xN array and return a Bx1x(N+1) array.
    """
    batch_shape = batch_of_samples.shape
    dims = len(batch_shape)

    if dims <= 0 or dims > 2:
        raise ValueError(
            f"reshape_and_concat received invalid sample shape {batch_shape}"
        )

    # Dim can only be 1 or 2 now
    if dims == 1:
        # Use vstack to turn this into a Bx1 stack of samples
        modified_samples = np.vstack(batch_of_samples)
    else:
        modified_samples = batch_of_samples

    batch_reshaped = modified_samples[:, np.newaxis, :]

    if storage_arr is None:
        return batch_reshaped

    return np.concatenate((storage_arr, batch_reshaped), axis=1)


def frame_to_step(frame_no, frame_skip=4):
    return -(-frame_no // frame_skip)


def simulate_batch_vec(episode_num, batch_size):
    simulation_ct = 0
    vec_env = gym.vector.make("ALE/Pong-v5", num_envs=parallelism)

    start = timeit.default_timer()

    xs_b, hs_b, dlogps_b, drs_b = (
        [],
        [],
        [],
        [],
    )

    reward_sum_b = []

    termination_steps_b = []

    while simulation_ct < batch_size:
        print(
            f"----Starting EPISODE {episode_num}, simulations {simulation_ct}...{simulation_ct+parallelism-1}"
        )

        observations, _ = vec_env.reset()  # Get the first game state
        prev_xs = None  # used in computing the difference frame

        xs_p, hs_p, dlogps_p, drs_p = (
            None,
            None,
            None,
            None,
        )
        reward_sum_p = np.zeros(parallelism)

        termination_frames_p = [None] * parallelism

        cur_steps = 0  # Debug relationship between frames and steps

        while True:
            cur_xs = np.array([prepro(obs) for obs in observations])

            # Calculate the difference frame (motion)
            xs = cur_xs - prev_xs if prev_xs is not None else np.zeros((parallelism, D))
            prev_xs = cur_xs

            # Run the forward pass of the policy network and sample an action
            policy_forward_out = [policy_forward(x) for x in xs]
            aprobs = np.array([aprob for aprob, _ in policy_forward_out])
            hs = np.array([h for _, h in policy_forward_out])
            actions = [
                2 if np.random.uniform() < aprob else 3 for aprob in aprobs
            ]  # In Pong, 3 is LEFT and 2 is RIGHT

            # Store various intermediate values used in backpropagation
            # TODO: Understand these variables
            xs_p = reshape_and_concat(xs_p, xs)
            hs_p = reshape_and_concat(hs_p, hs)
            ys = np.array(
                [1 if action == 2 else 0 for action in actions]
            )  # A "fake label"
            # grad that encourages the action that was taken to be taken
            # (see http://cs231n.github.io/neural-networks-2/losses if confused)
            loss_grad = ys - aprobs
            dlogps_p = reshape_and_concat(dlogps_p, loss_grad)

            # step the environment and get new measurements
            # Truncated is true if the game ends early
            cur_steps += 1
            observations, rewards, terminated, _truncated, infos = vec_env.step(actions)
            rewards = np.array(
                [
                    0 if tf else reward
                    for reward, tf in zip(rewards, termination_frames_p)
                ]
            )  # Zero out the post-termination rewards, so the reward sum is accurate

            reward_sum_p += rewards

            drs_p = reshape_and_concat(
                drs_p, rewards
            )  # record reward (has to be done after we call step() to get reward for previous action)

            # We need to maintain the end state
            if any(terminated):
                # At least one env terminated
                for idx, done in enumerate(terminated):
                    if done and termination_frames_p[idx] is None:
                        # Don't reset termination frames if it finishes again
                        frame_num = infos["final_info"][idx][
                            "episode_frame_number"
                        ]  # When this episode terminated
                        termination_frames_p[idx] = frame_num

                        ep_reward = reward_sum_p[idx]
                        win_ct = 21 if ep_reward > 0 else ep_reward + 21

                        print(
                            f"....env {simulation_ct+idx} terminated after {cur_steps} steps with total wins {win_ct}{'!!!!!' if win_ct > 0 else ''}"
                        )

            if all(termination_frames_p):
                print(f"..EPISODE#{episode_num}-SIMULATION#{simulation_ct} is DONE.")
                break

        # After episode ends, store the episodes in each batch
        xs_b.extend(xs_p)
        hs_b.extend(hs_p)
        dlogps_b.extend(dlogps_p)
        drs_b.extend(drs_p)
        reward_sum_b.extend(reward_sum_p)
        termination_steps_b.extend(
            [frame_to_step(f) for f in termination_frames_p]
        )  # Frameskip is 4, we want the ceiling to capture the last step

        # TODO: This means batch_size needs to be divisible by parallelism
        simulation_ct += parallelism

    end = timeit.default_timer()

    print(f"....time taken: {end-start}s")

    return xs_b, hs_b, dlogps_b, drs_b, reward_sum_b, termination_steps_b


def train_vec(
    model, cur_episode=0, grad_buffer=grad_buffer, rmsprop_cache=rmsprop_cache
):
    running_reward = None

    while True:
        # Next episode
        cur_episode += 1

        batched_results = simulate_batch_vec(
            episode_num=cur_episode, batch_size=batch_size
        )

        for xs, hs, dlogps, drs, reward_sum, termination_step in zip(*batched_results):
            # Sanity check just in case termination isn't round
            sample_len, _ = xs.shape

            if sample_len < termination_step:
                print(
                    f"Warning: sample count is {sample_len} but the termination step is {termination_step}"
                )

            # Don't use extra data, but include the last step
            xs = xs[:termination_step]
            hs = hs[:termination_step]
            dlogps = dlogps[:termination_step]
            drs = drs[:termination_step]

            # Stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)

            # compute the discounted reward backwards through time
            discounted_epr = discounted_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)

            # Calculate the gradient for this episode
            grad = policy_backward(epx, eph, epdlogp)
            for k in model:
                np.add(
                    grad_buffer[k], grad[k], out=grad_buffer[k], casting="unsafe"
                )  # accumulate grad over batch

            # Bookkeeping to keep tabs on the current rolling reward average
            running_reward = (
                reward_sum
                if running_reward is None
                else running_reward * 0.99 + reward_sum * 0.01
            )

            print(
                f"EPISODE#{cur_episode} reward is {reward_sum}. Running average is: {running_reward}"
            )

        # Backpropagate
        for k, v in model.items():
            g = grad_buffer[k]  # gradient
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        if cur_episode % 100 == 0:
            save_checkpoint("save.p", model=model, episode_num=episode_number)
            print(f"Saved model checkpoint at episode {episode_number}")


# TODO: Fix this initialization
def train(model, cur_episode=0, grad_buffer=grad_buffer, rmsprop_cache=rmsprop_cache):
    running_reward = None

    while True:
        # Next episode
        cur_episode += 1

        # Simulate the entire batch at once
        results = simulate_batch_mp(episode_num=episode_number, batch_size=batch_size)

        # Calculate the gradient on the batch
        for epx, eph, epdlogp, epr, reward_sum, thread_id in results:
            # compute the discounted reward backwards through time
            discounted_epr = discounted_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)

            # Calculate the gradient for this episode
            grad = policy_backward(epx, eph, epdlogp)
            for k in model:
                np.add(
                    grad_buffer[k], grad[k], out=grad_buffer[k], casting="unsafe"
                )  # accumulate grad over batch

            # Bookkeeping to keep tabs on the current rolling reward average
            running_reward = (
                reward_sum
                if running_reward is None
                else running_reward * 0.99 + reward_sum * 0.01
            )

            print(
                f"Reward for {thread_id} is: {reward_sum}. Running average is: {running_reward}"
            )

        # Backpropagate
        for k, v in model.items():
            g = grad_buffer[k]  # gradient
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        if cur_episode % 100 == 0:
            save_checkpoint("save.p", model=model, episode_num=episode_number)
            print(f"Saved model checkpoint at episode {episode_number}")


if __name__ == "__main__":
    train_vec(
        model,
        cur_episode=episode_number,
        grad_buffer=grad_buffer,
        rmsprop_cache=rmsprop_cache,
    )
