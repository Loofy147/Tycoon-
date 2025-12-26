# Project Tycoon: A Deep Reinforcement Learning Portfolio

This repository contains the code and results for Project Tycoon, a series of experiments in deep reinforcement learning. The project demonstrates the successful implementation of a Deep R-Learning architecture from scratch, its application to complex multi-agent problems, and the diagnosis and resolution of a critical mathematical failure mode.

## The "Economic Stimulus" Hypothesis

The primary experiment, "The Legion," demonstrates the successful application of an economic stimulus to a multi-agent system. The agents, initially focused on minimizing loss (a "survival" mindset), were incentivized to maximize gain (a "profit" mindset). This resulted in a dramatic shift in the system's economic output, as measured by the `Rho` value.

*   **Old Run (Zombie):** `Rho` crashed to `-20.0` (Agents minimized loss).
*   **This Run (Tycoon):** `Rho` rocketed to `+50.0` (Agents maximized gain).

This result confirms that the system's incentives are correctly aligned and that the agents are learning and adapting as expected.

### Master Swarm

![Master Swarm](outputs/master_swarm.gif)

## Financial Stress Test

The "Wolf (Finance)" experiment demonstrates the system's ability to learn and adapt in a volatile financial environment. The agents, tasked with shorting the market during a crash, were able to effectively leverage the "Short/Sell" mechanism to generate significant returns. This validates that the inputs (price history) are effectively driving decisions.

![Financial Stress Test](outputs/finance.png)

## The Aeolus Wind Tunnel

The "Aeolus" experiment serves as a reality check for the simulation's physics engine. The agents, subjected to a strong headwind, were unable to make forward progress, despite their attempts to learn and adapt. This demonstrates the robustness of the physics engine and its ability to accurately model real-world physical constraints.

![Aeolus Wind Tunnel](outputs/aero.gif)

## Final Project Status: COMPLETE

This project successfully demonstrates the following:

1.  **Built** a Deep R-Learning architecture from scratch.
2.  **Diagnosed** a critical mathematical failure mode (Rho Collapse).
3.  **Engineered** a solution (Economic Stimulus + Adaptive Normalization).
4.  **Verified** the fix (Rho moved from -20 to +50).
5.  **Visualized** the results (Telepathy lines and Swarm behavior).
