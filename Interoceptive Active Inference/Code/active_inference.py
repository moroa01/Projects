import numpy as np
from matplotlib import pyplot as plt
from pymdp import utils
from pymdp.agent import Agent

def create_agent(alpha=0.9, beta=0.9, zetaI=0.9, zetaE=0.9, chi=5.0):
    """
    - 1 observation 4 outcomes:
        0: Flower + Relaxed
        1: Flower + Aroused
        2: Spider + Relaxed
        3: Spider + Aroused
    - Hidden states:
        [0] external: Flower, Spider
        [1] internal: Diastole1, Diastole2, Systole
    """

    # Hidden states
    num_states = [2, 3]     # [external, internal]
    num_obs = [4]           # 4 combined outcomes

    # ===== A MATRIX =====
    # Likelihood A[o, s_ext, s_int]
    A = np.array([
        # (flower, relaxed)
        [
            [beta*alpha,          beta*alpha,          (1-beta)*0.5],   # ext = fiore
            [beta*(1-alpha),      beta*(1-alpha),      (1-beta)*0.5]    # ext = ragno
        ],
        # (flower, aroused)
        [
            [(1-beta)*alpha,      (1-beta)*alpha,      (beta)*0.5],
            [(1-beta)*(1-alpha),  (1-beta)*(1-alpha),  (beta)*0.5]
        ],
        # (spider, relaxed)
        [
            [beta*(1-alpha),      beta*(1-alpha),      (1-beta)*0.5],
            [beta*alpha,          beta*alpha,          (1-beta)*0.5]
        ],
        # (spider, aroused)
        [
            [(1-beta)*(1-alpha),  (1-beta)*(1-alpha),  (beta)*0.5],
            [(1-beta)*alpha,      (1-beta)*alpha,      (beta)*0.5]
        ]
    ], dtype=float)


    # ===== B MATRIX =====
    B = utils.obj_array(len(num_states))

    # External transitions B[0]: 2 x 2 x 1
    B[0] = np.array([
        [[zetaE], [1 - zetaE]],
        [[1 - zetaE], [zetaE]]
    ], dtype=float)

    # Internal transitions B[1]: 3 x 3 x 2 (actions: relaxed vs aroused)
    B[1] = np.array([
        [[(1 - zetaI) / 2, (1 - zetaI) / 2],
         [(1 - zetaI) / 2, (1 - zetaI) / 2],
         [zetaI, zetaI]],

        [[zetaI, (1 - zetaI) / 2],
         [(1 - zetaI) / 2, (1 - zetaI) / 2],
         [(1 - zetaI) / 2, (1 - zetaI) / 2]],

        [[(1 - zetaI) / 2, zetaI],
         [zetaI, zetaI],
         [(1 - zetaI) / 2, (1 - zetaI) / 2]]
    ], dtype=float)

    # ===== C MATRIX =====
    # Preferences on the 4 COMBINED outcomes
    C = utils.obj_array(1)
    C[0] = np.array([
        chi,      # Flower + Relaxed  (preferred)
        0.01,     # Flower + Aroused  (not preferred)
        0.01,     # Spider + Relaxed  (not preferred)
        chi       # Spider + Aroused  (preferred)
    ])

    # ===== D MATRIX =====
    # Priors over hidden states
    D = utils.obj_array(len(num_states))
    D[0] = np.array([0.5, 0.5])
    D[1] = np.array([0.33333333333, 0.333333333333, 0.3333333333333])

    # ===== E MATRIX =====
    # Habits / prior policies
    E = np.array([10/20, 10/20])

    # Agent controlling only the internal state factor (index 1)
    agent = Agent(A=A, B=B, C=C, D=D, E=E, control_fac_idx=[1])
    return agent

def simulation(agent = create_agent(1, 1, 1, 1, 1), preE = 0, preI = 0, num_iterations = 8, verbose=False):
    battiti = []
    visione = []
    osservazioni = []
    aspettativa = []
    azioni = []
    energy = []

    sI = ["Diastole", "Diastole2", "Systole"]
    sE = ["Flower", "Spider"]
    a = ["Relaxed", "Aroused"]
    b = ["Flower Relaxed", "Flower Aroused", "Spider Relaxed", "Spider Aroused"]
    np.set_printoptions(suppress=True, precision=3)

    for t in range(num_iterations):       
        battiti.append(preI)
        visione.append(preE)
        observation = utils.sample(agent.A[0][:, preE, preI])
        
        osservazioni.append(observation)

        qs = agent.infer_states([observation])

        B_extero = np.squeeze(agent.B[0])
        pred_next = B_extero @ qs[0]

        aspettativa.append(pred_next)
        s0 = np.argmax(qs[0])
        s1 = np.argmax(qs[1])

        poli = agent.infer_policies()
        energy.append(poli[1])
        
        action = agent.sample_action()
        action = [int(action[0]), int(action[1])]

        azioni.append(action[1])

        external_state = utils.sample(agent.B[0][:, preE])
        internal_state = utils.sample(agent.B[1][:, preI, action[1]])
        
        if verbose:
            print("Iteration ", t, ":")
            print("Hidden states:\t", sE[preE], "\t\t", sI[preI])
            print("Observation:\t", b[observation], "\n")
            print("\033[1mInfer. states:\033[0m\t", qs[0], "\t\t", qs[1])
            print("Policies: ", poli, "\n")
            print("Action: ", action)
            print("My expectations:", pred_next)
            print(f"\nHidden states = {preE, preI} --> {external_state, internal_state}")
            print(f"Hidden states = {sE[preE], sI[preI]} --> {sE[external_state], sI[internal_state]}")
            print("\033[1m-------------------------------------------------------------\033[0m")
        preE = external_state
        preI = internal_state
    return [battiti, visione, osservazioni, aspettativa, azioni, energy]

def forced_simulation(agent = create_agent(1, 1, 1, 1, 1), heart_state = 0, iterations=np.zeros(8).astype(int), verbose=False):
    battiti = []
    visione = []
    osservazioni = []
    aspettativa = []
    azioni = []
    energy = []

    sI = ["Diastole", "Diastole2", "Systole"]
    sE = ["Flower", "Spider"]
    a = ["Relaxed", "Aroused"]
    b = ["Flower Relaxed", "Flower Aroused", "Spider Relaxed", "Spider Aroused"]
    np.set_printoptions(suppress=True, precision=3)
    t=1

    for external_state in iterations:
        battiti.append(heart_state)
        visione.append(external_state)
        observation = utils.sample(agent.A[0][:, external_state, heart_state])
        
        osservazioni.append(observation)
       
        qs = agent.infer_states([observation])

        B_extero = np.squeeze(agent.B[0])  # shape (2,2)
        pred_next = B_extero @ qs[0]

        aspettativa.append(pred_next)
        s0 = np.argmax(qs[0])
        s1 = np.argmax(qs[1])

        poli = agent.infer_policies()
        energy.append(poli[1])
        
        action = agent.sample_action()
        action = [int(action[0]), int(action[1])]

        azioni.append(action[1])

        internal_state = utils.sample(agent.B[1][:, heart_state, action[1]])
        ddd = utils.sample(agent.B[0][:, external_state])
        
        if verbose:
            print("Iteration ", t, ":")
            print("Hidden states:\t", sE[external_state], "\t\t", sI[heart_state])
            print("Observation:\t", b[observation], "\n")
            print("Infer. states:\t", sE[s0], "\t\t", sI[s1])
            print("Policies: ", poli, "\n")
            print("Action: ", action)
            print(f"\nHidden states = {external_state, heart_state} --> forced, {internal_state}")
            print(f"Hidden states = {sE[external_state], sI[heart_state]} --> forced, {sI[internal_state]}\n")
            print("-------------------------------------------------------------")

        t += 1
        heart_state = internal_state
    return [battiti, visione, osservazioni, aspettativa, azioni, energy]

def simulation_with_sequence(agent = create_agent(0.9, 0.9, 0.9, 0.9, 0.9), heart_state = 0, sequence=np.zeros(15).astype(int), iterations=80, verbose=False):
    battiti = []
    visione = []
    osservazioni = []
    aspettativa = []
    azioni = []
    energy = []

    sI = ["Diastole", "Diastole2", "Systole"]
    sE = ["Flower", "Spider"]
    a = ["Relaxed", "Aroused"]
    b = ["Flower Relaxed", "Flower Aroused", "Spider Relaxed", "Spider Aroused"]
    np.set_printoptions(suppress=True, precision=3)

    length = len(sequence)

    internal_state = heart_state
    external_state = 0

    for i in range(iterations):
        
        if i < length:
            external_state = sequence[i]
        else:
            external_state = utils.sample(agent.B[0][:, external_state])
        
        battiti.append(internal_state)
        visione.append(external_state)

        observation = utils.sample(agent.A[0][:, external_state, internal_state])
        
        osservazioni.append(observation)
       
        qs = agent.infer_states([observation])

        B_extero = np.squeeze(agent.B[0])
        pred_next = B_extero @ qs[0]

        aspettativa.append(pred_next)
        s0 = np.argmax(qs[0])
        s1 = np.argmax(qs[1])

        poli = agent.infer_policies()
        energy.append(poli[1])
        
        action = agent.sample_action()
        action = [int(action[0]), int(action[1])]

        azioni.append(action[1])

        next_int = utils.sample(agent.B[1][:, internal_state, action[1]])
        
        if verbose:
            print("Iteration ", i, ":")
            print("Hidden states:\t", sE[external_state], "\t\t", sI[internal_state])
            print("Observation:\t", b[observation], "\n")
            print("Infer. states:\t", sE[s0], "\t\t", sI[s1])
            print("Policies: ", poli, "\n")
            print("Action: ", action)
            print(f"\nHidden states = {external_state, internal_state} --> forced, {next_int}")
            print(f"Hidden states = {sE[external_state], sI[internal_state]} --> forced, {sI[next_int]}\n")
            print("-------------------------------------------------------------")

        internal_state = next_int
    return [battiti, visione, osservazioni, aspettativa, azioni, energy]

def plot_simulation_results(battiti, visione, osservazioni, aspettativa, azioni, energy):
    iterations = range(len(battiti))

    stato_interno = ["Diastole", "Diastole2", "Systole"]
    stato_esterno = ["Flower", "Spider"]
    osservazione_est = ["Obs Flower", "Obs Spider"]
    osservazione_int = ["Relaxed", "Aroused"]
    a = {0: [0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]}
    
    penso = [a[x][0] for x in osservazioni]
    feel = [a[x][1] for x in osservazioni]

    fig, ax = plt.subplots(6, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios':[0.4, 0.75, 0.4, 0.4, 0.75, 0.75]})

    # External state
    ax[0].plot(iterations, visione, marker='o', linestyle='-')
    ax[0].set_yticks([0,1])
    ax[0].set_yticklabels(stato_esterno)
    ax[0].set_title("External state")

    # Internal state
    ax[1].plot(iterations, battiti, marker='o', linestyle='-')
    ax[1].set_yticks([0,1,2])
    ax[1].set_yticklabels(stato_interno)
    ax[1].set_title("Internal state")

    # External observation
    ax[2].plot(iterations, penso, marker='x', linestyle='--', color='orange')
    ax[2].set_yticks([0,1])
    ax[2].set_yticklabels(osservazione_est)
    ax[2].set_title("External observation")

    # Internal observation
    ax[3].plot(iterations, feel, marker='x', linestyle='--', color='red')
    ax[3].set_yticks([0,1])
    ax[3].set_xticks([x for x in range(len(iterations))])
    ax[3].set_yticklabels(osservazione_int)
    ax[3].set_title("Internal observation")

    zeros = [a[0] for a in aspettativa]
    ones = [a[1] for a in aspettativa]

    # Probability of expectation between on next iteration
    ax[4].plot(iterations, zeros, marker='o', linestyle='-', color="#09FF09", label="P(Flower)")
    ax[4].plot(iterations, ones, marker='o', linestyle='-', color="black", label="P(Spider)")
    ax[4].set_yticks([0,1])
    ax[4].set_title("Expectation")
    ax[4].legend()

    # Chosen action and relative free energy
    ax[5].plot(iterations, azioni, marker='o', linestyle='-')
    ax[5].set_yticks([0,1])
    ax[5].set_yticklabels(osservazione_int)
    ax[5].set_title("Chosen action with free energy")
    ax[5].set_xlabel("Iteration")
    ax[5].axhline(y=0.5, linestyle="--", linewidth=0.75, color="black")
    
    for i, v in enumerate(energy):
        ax[5].text(i, 0.43 + 0.18, f"{v[0]:.2f}", ha="center", va="bottom")
        ax[5].text(i, 0.43 - 0.2, f"{v[1]:.2f}", ha="center", va="bottom")

    plt.tight_layout()

    plt.show()
