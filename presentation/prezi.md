---
title: Parallel Enhanced Whale Optimization Algorithm
theme: simple
author: Bevan Stanely
---

## Parallel Enhanced Whale Optimization Algorithm

By Bevan Stanely https://bevs.xyz/

---

## Meta-Heuristic Optimization Algorithms

- Rely on simple concepts
- No need for gradient information
- Can bypass local optima
- Useful across wide range of problems

Note: Motivation

Popular in Engineering applications because-

Stochastic hence need random numbers

Exploratory and Exploitatory phases

---
<!-- .slide: data-background-image="./img/humbback.jpg" -->
## Whale Optimization Algorithm (WOA)<!-- .element: style="color:white;background-color: blue;" -->

- Iterative<!-- .element: style="color:white;background-color: black;"-->
- Swarm Based<!-- .element: style="color:white;background-color: black;"-->

Note: Inspired by the bubble-net feeding behavior of humpback whales

----

## Drawbacks

- Low exploration ability
- Slow convergence speed
- Sticking with local solution easily.

WOAmM

Note:

Enhanced Whale Optimization Algorithm

Addresses these drawbacks

---

We want to develop a Parallel WOAmM with SpeedUPs

- Quality and size of RNGs
- Data dependencies

Note:

- GPU RNGs have lower quality host implementations are bulky and slow
- Sequential algorithms have data dependencies which aid in parallelization

---

## Enhanced Whale Optimization Algorithm (WOAmM)

Note:

Hybrid algorithm with

Two components

----

## mSOS

Modified Mutualism phase of the symbiotic organism search (mSOS) algorithm

$$
P^{(k+1)}_i= P^{(k)}_i+rnd\cdot(P_s - MV\cdot BF1)
$$

$$
P^{(k+1)}_r= P^{(k)}_n+rnd\cdot(P_s - MV\cdot BF2)
$$

<!-- .element: class="fragment"-->

Note:

- Pick two position vectors from other threads
- Pr = max() and Ps = min()
- BF = 1 or 2
- MV mean(pi,ps)

----

## WOA

![](img/bubble-net.png)

---

## How to Parallelise?

- Model individuals as GPU threads
- Intra-warp Communication with Butterfly reduction
- Avoiding Warp Divergence
- Random Numbers

```C
for(each thread in warp) do{
    while(k < max_iter){
        mSOS();
        WOA();
        k++;
    }
}

```

Note:

- Intra warp communication is faster than shared memory
- If else conditions have been avoided by using pointer arrays
- Random numbers: Device RNGs and Host RNG

---

## Experiment

| Parameters | Range                                  |
| ---------- | -------------------------------------- |
| RNG        | MTGP32,MRG32k3a,Philox_4x32_10         |
| Iterations | {30,100,300}                           |
| Blocks     | {1,2,4,6}                              |
| Functions  | {Sphere,Rosenbrock,Rastrigin,Griewank} |

Function Properties:

- Dimension = 30
- Optimum Value = 0

Note:

Sequential algorithm used Mersanne Twister 64 bit RNG

We have ignored the RNG initiation times for the experiments and focussed on computation times and memory copy costs.

It is hence suitable for bulk optimizations, so that reuse of state is possible.

---

![](./img/fitness.png)

Note:

The parallel implementation across all parameters was compared with sequential algorithm for their optimization quality.

---

![](./img/speed_up.png)

MRG32k3a with 100 iterations and 2 blocks appears to give best results<!-- .element: class="fragment"-->

Note:

The parallel implementation across all parameters was compared with sequential algorithm for their speedUp.

---

- Chaotic Maps instead of RNG

---

## ThanK U

