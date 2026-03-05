# Exercise 2.11: Channel Estimation using LDAMP and LISTA-CE

**References & Resources:**
* **LDAMP:** [Reference Code](https://github.com/hehengtao/LDAMP_based-Channel-estimation)
* **LISTA-CE:** [Reference Code](https://github.com/King-SmallA/LISTA-CE) | [Training Data](https://drive.google.com/drive/folders/1OeRStZSpSX7V3PTSwgqUjdhIQ6qlW3dG)

This repository provides the starter code for Exercise 2.11. Your task is to complete the core implementations of both the **LISTA-CE (Python)** and **LDAMP (MATLAB)** algorithms for channel estimation.

After completing the code, you will evaluate their performance under different channel sparsity levels, adjust key hyperparameters, and analyze their impact on estimation accuracy (NMSE) and convergence speed.

## What You Need to Do

| Step | Task | Details |
| :---: | :--- | :--- |
| 1 | **Code Completion** | Open `LISTA-CE.py` and `main_new_old.m` (located in the `LDAMP_based-Channel-estimation-master` folder). Search for the `# TO-DO` (or `% TO-DO`) blocks and complete the missing code based on the instructions. |
| 2 | **Run & Test** | Execute `LISTA-CE.py` (Python) and `main_new_old.m` (MATLAB) to ensure your implementations run without errors and generate baseline results. |
| 3 | **Parameter Tuning** | Follow the [Experimentation Guide](#experimentation-guide) below to adjust parameters such as channel sparsity, network architecture, and learning rate. |
| 4 | **Observe & Analyze**| Compare the NMSE vs. SNR curves and convergence speeds. Analyze how different parameter settings impact the channel estimation performance. |

## File Structure

| File / Directory | Purpose |
|------|---------|
| `LISTA-CE.py` | Python starter script for the LISTA-CE algorithm (requires TensorFlow/PyTorch). Contains TO-DOs for the ISTA update step, loss function, and NMSE. |
| `wcmlbook/ch2/Exercise_2.11/LDAMP_based-Channel-estimation-master/` | Directory containing the MATLAB files for the LDAMP algorithm. *(Note: You may need to extract the `SCAMPI-MATLAB.7z` archive located here first).* |
| └── `scampi-vs-ssd/main_new_old.m` | MATLAB starter script for the LDAMP algorithm. Contains TO-DOs for compressive measurement, D-AMP solver initialization, and NMSE plotting. |

---

## Detailed Task Breakdown

### Part 1: Code Completion

**For `LISTA-CE.py` (Python):**
*   **TO-DO 1:** Implement the core ISTA update step in the `ista_block`.
*   **TO-DO 2:** Define the Mean Squared Error (MSE) loss function in `compute_cost`.
*   **TO-DO 3:** Calculate the Normalized Mean Square Error (NMSE) in `run_vali`.
*   **TO-DO 4:** Define the training optimizer (e.g., AdamOptimizer) in the main block.

**For `main_new_old.m` (MATLAB):**
*   **TO-DO 1:** Perform the compressive measurement using the `MultSeededHadamard` operator.
*   **TO-DO 2:** Calculate the noise variance (`nvar`).
*   **TO-DO 3:** Call the D-AMP solver (`DAMP_SNR`) to reconstruct the signal.
*   **TO-DO 4:** Calculate the normalized squared error for the algorithms.
*   **TO-DO 5:** Plot the NMSE vs. SNR curves using `semilogy`.

---

## Experimentation & Analysis Guide

Once the code is complete, perform the following experiments:

### A. Compare Performance under Different Channel Sparsity Levels
Adjust the parameter controlling the number of non-zero paths in the channel to see how sparsity affects performance:
*   **LISTA-CE (Python):** Adjust the `--Path_num` command-line argument (e.g., `default=3`).
*   **LDAMP (MATLAB):** Adjust the variable `L` in the script (e.g., `L = 3` for sparse, `L = 10` for denser).
*   *Metrics to evaluate:* Final estimation accuracy (NMSE vs. SNR) and convergence speed (iterations required to reach a stable NMSE).

### B. Analyze the Impact of Algorithm Parameters
Modify the following hyperparameters and assess their effects on estimation accuracy:

**1. Network Architecture (Depth):**
*   **Python:** Adjust the `--layers` command-line argument to change the network depth.
*   **MATLAB:** Adjust `n_DnCNN_layers` to use a different pre-trained denoiser depth (e.g., 17 vs. 20).

**2. Learning Rate & Training Stability:**
*   **Python:** Modify the `lr` variable in the script to observe its effect on training stability and final performance.
*   **MATLAB:** *(Note: Learning rate is not adjustable in the provided inference script, as it pertains to the offline training of the DnCNN denoiser).*

**3. Training / Algorithm Iterations:**
*   **Python:** Modify the `max_episode` variable to study the trade-off between underfitting and overfitting.
*   **MATLAB:** Adjust `AMP_iters` to study the convergence behavior and its impact on final accuracy.