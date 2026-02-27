# Exercise 6.6: FedAvg Implementation under Packet Loss

This repository provides the starter code for Exercise 6.6. Your objective is to simulate the impact of **packet loss** on federated learning using the FedAvg algorithm.


## Experiment Setup
The script is pre-configured with the specific parameters from the textbook:
* **Clients $N$ :** 100 (10% sampled per communication round)
* **Communication Rounds $T$ :** 100
* **Local Epochs $E$ :** 5
* **Non-i.i.d. Data:** Dirichlet distribution ($\alpha = 0.5$)
* **Packet Loss Rates:** 1%, 5%, and 10%

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `exercise_6.6_starter.py` and complete the `# YOUR CODE HERE` blocks:<br>  • **`unreliable_channel`**: Drop each update with probability `loss_rate`.<br>  • **`client_train`**: Zero gradients, forward pass, compute NLL loss, `backward()` and `opt.step()`. |
| **Run** | Execute: `python exercise_6.6_starter.py` |
| **Observe** | A plot `packet_loss_impact.png` is saved. Observe that 1% loss mirrors standard FedAvg, while 5-10% slows convergence and degrades accuracy. |

> **Hint:** Use `np.random.rand() > loss_rate` in `unreliable_channel` to determine if a client's update successfully reaches the server.

## Files

| File | Purpose |
|------|---------|
| `exercise_6.6_starter.py` | Starter script (with TODO). |
| `packet_loss_impact.png` | Generated after you run the script. |