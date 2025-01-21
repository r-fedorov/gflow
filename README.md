# OER Catalyst Discovery with GFlowNets

This repository presents a solution for exploring the chemical space of High Entropy Oxides to identify optimal catalysts for the Oxygen Evolution Reaction. The implementation focuses on an MVP approach while laying the groundwork for future improvements and extensions.

---

## Project Overview

The goal of this project is to develop an active learning  algorithm based on **GFlowNets** to efficiently search for the optimal catalytic performance of HEOs. The chemical space is defined by a mixture of five different metal oxides, forming a 5-dimensional simplex where the sum of all component percentages equals 100%.

1. **MVP Solution – ContinuousCube Environment:**
   - Implements state representation for metal compositions \( A, B, C, D \), and \( E \) such that \( E = 100\% - \) sum(\( A \) to \( D \)).
   - Enforces simplex constraints with renormalization if \( E \) drops below 10%.
   - Includes a penalty term for invalid states, factored into the output of a predictive proxy model.
   - Utilizes pre-computed values as described in the task.

2. **Proxy Function:**
   - Combines predictions from two retrained ML models (from the paper) with penalty energy adjustments to compute a reward.


`````python
Project Root
├── README.md             # Project documentation
├── buffer                # Provided data converted to A, B, C, D format
├── logs                  # Experimental results and logs
├── mars_configs          # .yaml files and Hydra configuration
├── retraining_models     # Data and weights for models retrained as per the provided paper
├── buffer_energy.py      # Handles buffering of provided data without sampling from the proxy
├── data2state.ipynb      # Converts metal ratios to A, B, C, D, (E) format and back
├── main.py               # Main entry point of the project
├── mvp_mars_env.py       # MVP environment implementation
├── mvp_proxy.py          # Proxy function: 2 MLPs with penalty handling
├── gflownet_on_mars.py   # Main GFlowNet agent implementation
└── robo_models.py        # Code for MLPs used in the project
`````





3. **Bonus Points – SimplexWalker (Demo):**

Although this solution provides a functional implementation, there is significant room for optimization and enhancement. The current approach serves as a robust starting point, with a clear path for further development.

#### UDPDATE 

fixed the log prob, now it works. Small test: Train data Mean Score -> 1.8514679012345678 SimplexWalker -> 2.20 
#### UDPDATE #2

Added a random and deafault param configs, now can be run from a main function and is being parametrised by its own output

**Outline of the Advanced SimplexWalker Environment**

The SimplexWalker environment introduces a more sophisticated approach to navigating the chemical space. The current implementation is functional for sampling, additional debugging is required to resolve issues with computing log probabilities.

Key Features of SimplexWalker:

Two Operating Modes:

Mode 1: Samples states for A to D, ensuring their sum is less than or equal to 1, and calculates E as the remaining budget.
Mode 2: Iteratively samples all five dimensions until the budget is fully allocated (i.e., reaches 0).

Enhanced Sampling with OneHotConditional:

A added distribution, OneHotConditional, determines which dimension is updated at each step by the mixture of the Beta Distributions.

A minimal increment is applied exclusively to the selected dimension.

The realtive Movment is computed exactly the same as in contious cube, but the avaible room is denoted as 1 - sum(states), adn then the one-hot applies.

End-of-Sequence (EOS) Trigger:

The process concludes when the sum of the states reaches 1 (the total computational budget). (or earlier, if we are sampling to simplex and not on simplex)

Log Probability Adjustments:

The log probabilities are modified to reflect the changes accurately:

Log probabilities are masked for dimensions where no updates occur.

The log probability of the one-hot selection is added.










