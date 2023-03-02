import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def animate(frames):
    """Animate Render"""
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for frame in frames:
        im = ax.imshow(frame, animated=True)
        # if i == 0:
        #     ax.imshow(frame)  # show an initial one first
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    # To save the animation, use e.g.

    ani.save("rollout.gif")
    plt.show()


def plot(results: dict[str, tuple]):
    """Plot training results"""
    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    n_policies = len(results)

    # Episode Length and Episode Return over Episodes
    # ------------------------------------------------------------------
    fig = plt.figure()
    axes = fig.subplots(ncols=1, nrows=2, sharex=True, sharey="row")
    for i in range(n_policies):
        policy_name = list(results.keys())[i]
        ret = results[policy_name]
        visited_positions, losses, cum_rewards, env, actions, dones = ret

        axes[0].plot(env.length_queue, label=policy_name, marker="o")
        axes[0].set_ylabel("episode length")
        axes[1].plot(env.return_queue, label=policy_name, marker="o")
        axes[1].set_ylabel("return")
        axes[1].set_xlabel("episode")
        axes[0].legend()
    fig.set_tight_layout(True)
    plt.show()

    #  Loss over Steps
    # ------------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(n_policies):
        policy_name = list(results.keys())[i]
        ret = results[policy_name]
        visited_positions, losses, cum_rewards, env, actions, dones = ret

        steps, loss = zip(*losses)

        ax.plot(steps, loss, label=policy_name)
    ax.set_ylabel("loss")
    ax.set_xlabel("step")
    ax.legend()
    fig.set_tight_layout(True)
    plt.show()

    # Actions over Steps
    # ------------------------------------------------------------------
    # Episode endings are marked with a faint vertical line.
    fig = plt.figure(figsize=(12, 6))
    axes = fig.subplots(nrows=n_policies, ncols=1, sharex=True, sharey=True)
    for i in range(n_policies):
        policy_name = list(results.keys())[i]
        ret = results[policy_name]
        visited_positions, losses, cum_rewards, env, actions, dones = ret

        steps, action = zip(*actions)
        steps, done = zip(*dones)
        where_done = np.where(done)[0]

        axes[i].plot(
            steps,
            action,
            label=policy_name,
            marker=".",
        )
        axes[i].set_ylabel("action")
        axes[i].set_title(policy_name)
        axes[i].set_yticks(np.arange(0, env.action_space.n))
        axes[i].set_yticklabels([env.actions(i).name for i in range(env.action_space.n)])
        axes[i].set_xlabel("step")
        for vline in where_done:
            axes[i].axvline(vline, color="k", alpha=0.2)
    ax.legend()
    fig.set_tight_layout(True)
    plt.show()

    # Actions as Histogram
    # ------------------------------------------------------------------
    actions_ = []
    for policy_name, ret in results.items():
        visited_positions, losses, cum_rewards, env, actions, dones = ret
        steps, A = zip(*actions)
        actions_.append(pd.DataFrame({"policy": [policy_name] * len(A), "action": A}))
    df_actions = pd.concat(actions_)

    ax = sns.histplot(data=df_actions, x="action", hue="policy", multiple="dodge")
    ax.set_xticks(np.arange(0, env.action_space.n))
    ax.set_xticklabels([env.actions(i).name for i in range(env.action_space.n)])
    plt.show()

    # Visitied Positions on the Grid
    # ------------------------------------------------------------------
    inches = 6
    fig = plt.figure(figsize=(inches, n_policies * inches))
    axes = fig.subplots(ncols=n_policies, nrows=1, sharex=True, sharey="row")
    for i in range(n_policies):
        policy_name = list(results.keys())[i]
        ret = results[policy_name]
        visited_positions, losses, cum_rewards, env, actions, dones = ret

        print(visited_positions)

        visited_positions_norm = visited_positions / np.sum(visited_positions)

        axes[i] = sns.heatmap(visited_positions_norm, ax=axes[i])
        axes[i].set_title(policy_name)
        axes[i].set_aspect("equal")  #
    fig.set_tight_layout(True)
    plt.show()
