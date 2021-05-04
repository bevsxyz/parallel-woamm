---
title: Parallel Enhanced Whale Optimization Algorithm
author: Bevan Stanely 17285
date: 30 April 2021
classoption: twocolumn
autoEqnLabels: true
bibliography: ref.bib
link-citations: true
color-links: true
linkcolor: blue
urlcolor: blue
citecolor: blue
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---

## Aim

To develop a parallel GPU implementation of the recently proposed Enhanced Whale Optimization Algorithm(WOAmM) and find speed-ups with benchmark functions.

## Background

The Whale Optimization Algorithm (WOA) is a meta-heuristic optimization algorithm developed by @mirjaliliWhaleOptimizationAlgorithm2016. It uses the humpback whales' hunting mechanism. Like most optimization algorithms, it works by spawning a population of agents and engaging them in exploration and exploitation phases. The exploratory phase helps to explore the search space extensively, whereas the exploitatory step refines promising solutions from the exploratory stage.  Though effective, WOA may suffer from the low exploration of the search space. An enhanced WOA (WOAmM), proposed recently by @chakrabortyNovelEnhancedWhale2021, adds a modified mutualism phase of Symbiotic Organisms Search (SOS) to WOA to improve the exploration.

Optimization problems are essential in engineering and science research, and parallelization of the same is of practical use.

## Enhanced Whale Optimization Algorithm(WOAmM)

A brief overview of the WOAmM and its component optimization algorithms follows. The focus is on the mathematical equations required for the computations. Refer to Algorithm 1 for the pseudo-code.

### Whale Optimization Algorithm

The whole search process is divided into three phases, searching the prey, encircling the target, and spiral bubble-net feeding maneuver.

#### Searching the prey

$$
\overline{D}= |C\cdot P^{(k)}_{rnd}-P^{(k)}|
$$
$$
P^{(k+1)}=  P^{(k)}_{rnd}-A\cdot \overline{D}
$$

where $P$ indicate a position vector of the population, $P_{rnd}$ is a vector
which is chosen randomly from the present population, the current
iteration is denoted by $k$, $\overline{D}$ is the distance between the random and
current individual of the population, the dot $(\cdot)$ operator indicates the
element by element multiplication process and $| |$ is used to define
absolute value.

The two parameters, known as coefficient vectors, $A$, and $C$ are
calculated as follows:

$$A = 2a_1 \times rnd - a_1$$

$$C = 2 \times rnd$$

where the value of $a_1$ is a number decreased linearly from 2 to 0 during
iteration and $rnd$ is a random number within the interval.

#### Encircling the prey

$$
\overline{D}= |C\cdot P^{(k)}_{best}-P^{(k)}|
$$

$$
P^{(k+1)}= P^{(k)}_{best} A\cdot \overline{D}
$$

Where $P_{best}$ is the current best solution.

#### Bubble-net attacking strategy

$$
D^*= P^{(k)}_{best}-P^{(k)}
$$

$$
P^{(k+1)}= D^*\cdot e^{bl} \cdot \cos{2\pi l} + P^{(k)}_{best}
$$

where $b$ is used to defining the shape of the logarithmic spiral, and the
value of $b$ is a constant, $l$ is a random number calculated according to the
following equation:

$$
l = (a_2 - 1)rnd + 1
$$

where the value of $a_2$ is decreased linearly within the value (-1) to (-2)
during the iteration process, and $rnd$ is an arbitrary number in the range
[0,1].

During the search process, switching between the exploration and
exploitation process is chosen depending on the value of $|A|$. If $A| \geq 1$,
the exploration process comes into existence, enabling global search
using Eq. 1 and 2. When $|A| < 1$, then updating the positions of individuals is performed by using eq. 6 or 8. Depending on a probability
value $\beta$, which is 0.5 for each strategy WOA process switches between
encircling prey or bubble-net attacking strategy.

\begin{algorithm}
\SetAlgoLined
 Initialize the population $P_i(i = 1,2\ldots n)$\;
 Calculate the fitness value of each \textit{search-agent}.\;
 Find the best \textit{search-agent} $P_{best}$\;
 Initialize $k = 0 \And \max_{iter}$\;
 \While{$k \leq \max_{iter}$}{
    \For{every \textit{search-agent}}{
        Select two other \textit{search-agent} $(P_m \And P_n)$ randomly where $P_i \neq P_m \neq P_n$\;
        \eIf{fitness ($P_m$)< fitness ($P_n$)}{
            Calculate the new value of $P_i \And P_n$ using eq.10 and eq.11 respectively.
        }{
            Calculate new value of $P_i \And P_n$ using eq.12 and eq.13 respectively.
        }
        Calculate the fitness of $P_i^{k+1}$ and $P_m^{k+1}$ or $P_n^{k+1}$\;
        Update the value of $P_i$ and $P_n$ or $P_m$ if the new fitness value is minimum (for minimization problem).\;
    }
    Calculate $P_{best}$\;
    \#Procedure WOA starts from here\;
    \For{every \textit{search-agent}}{
        Update $A,C,l \And \beta$\;
        \eIf{$\beta < 0.5$}{
            \eIf{$|A| \geq 1$}{
                Select a random \textit{search-agent} $P_{rnd}$\;
                Update the position of current \textit{search-agent} by eq.2
            }{
                Update the position of current \textit{search-agent} by eq.6
            }
        }{
            Update the position of current \textit{search-agent} by eq.8
        }
        Check boundary conditions\;
    }
    $k=k+1$
 }
 Return $P_{best}$
 \caption{Pseudo-code of the WOAmM algorithm.}
\end{algorithm}

#### Modified Mutualism phase of symbiotic organism search (SOS) algorithm

In this phase for every individual $(P_i)$, two random in­dividuals 
$(P_m \And P_n)$ are selected from the population where $i \neq m \neq n$.
The individual with minimum fitness among these two randomly chosen
individuals is chosen to enumerate the new value of the present indi­vidual $P_i$ and the other random individual. 
If $P_m$ is the individual with
minimum fitness between the two, the updating process is as follows:

$$
P^{(k+1)}_i
= P^{(k)}_i + rnd(0, 1)\times(P_m - MV\times BF1)
$$

$$
P^{(k+1)}_n
= P^{(k)}_n + rnd(0, 1)\times(P_m - MV\times BF2)
$$

Otherwise,

$$
P^{(k+1)}_i
= P^{(k)}_i + rnd(0, 1)\times(P_n - MV\times BF1)
$$

$$
P^{(k+1)}_m
= P^{(k)}_m + rnd(0, 1)\times(P_n - MV\times BF2)
$$

where $rnd$ is a random number
with a uniform distribution between [0, 1], $MV$ is the mutual vector,$BF$
is the benefit vector, $k$ is the generation. $MV$ and $BF$ are calculated as
follows:

$$
MV = \frac{P_i+P_j}{2}
$$

where, $j=n$ in first occasion and $j=m$ in second occassion.

$$
BF = round(1+rnd)
$$

The round function is used to set the value of $BF$ as one or two. $BF$ is
used to identify whether an organism partially or fully benefits from the
interaction among individuals from the population.

## Strategy

From the pseudo-code available under Algorithm 1, we can identify regions amenable for parallelization. The outer while loop
 has to be sequential, and hence there is no scope for parallel execution. Our next bet would be the two inner `for` loops
 corresponding to the modified mutualism phase of SOS and WOA, respectively. However, there is one caveat if we choose to
 parallelize these regions. Both component optimization algorithms update the individuals with a random individual selected
 from the population. The probability that a randomly selected individual has already been updated increases from 0 for the
 first individual to 1 for the last individual in serial execution. If we parallelize the algorithm overlooking this issue,
 we will end up reducing the stochasticity of exploration. We will revisit the problem in the end.

### The plan for a single block

A single block will execute the whole algorithm with threads equal to the population size. A pseudo-code is available under Algorithm 2.

1. Each thread will execute commands 1 to 4 in parallel. (Will store the population values in a shared array.
2. Inside the while loop, all threads synchronize the shared array locally.  ( A modulo-based for loop is used to avoid bank conflicts. Another option is the Cuda shuffle all reduction for minimum)
3. Then, each thread executes the local population array's optimization codes except that there would be a sync event before WOA.
4. Loop.
5. After the while loop ends, the device will return the best solution to the host.

\begin{algorithm}
\SetAlgoLined
 \# Start kernel from here for single warp\;
 Initialize the population $P_i(i = 1,2\ldots n)$ in shared memory array\;
 Calculate the fitness value of each \textit{search-agent}.\;
 Find the best \textit{search-agent} $P_{best}$\;
 Initialize $k = 0 \And \max_{iter}$\;
 Copy the $P_i$ shared array to a local array\;
 \While{$k \leq \max_{iter}$}{
    \For{thread}{
        Select two other \textit{search-agent} $(P_m \And P_n)$ randomly where $P_i \neq P_m \neq P_n$\;
        Initialize array of indexes ind = $[P_m,P_n]$\;
        Set int index = $P_m < P_n$\;
        Calculate the new value of $P_i \And population[ind[index]]$ using eq.10 and eq.11\;
        Calculate the fitness of $P_i^{k+1}$ and $P_m^{k+1}$ or $P_n^{k+1}$\;
        Update the value of $P_i$ and $P_n$ or $P_m$ if the new fitness value is minimum (for minimization problem).\;
    }
    \# Maybe avoiding the following sync can improve stochasticity\;
    Syncronize local array with shared array use min value of $P_i$ or the shuffle cuda operation could be faster\;
    Calculate $P_{best}$\;
    \#Procedure WOA starts from here\;
    \For{every \textit{search-agent}}{
        Update $A,C,l \And \beta$\;
        \eIf{$\beta < 0.5$}{
            Array = $[P_{rnd},P_{best}]$
            Index int ind = $|A| \geq 1$
            Update the position of current \textit{search-agent} by eq.2
        }{
            Update the position of current \textit{search-agent} by eq.8
        }
        Check boundary conditions\;
    }
    Syncronize local array with shared array use min value of $P_i$ or the shuffle cuda operation could be faster\;
    $k=k+1$
 }
 Return $P_{best}$
 \caption{Pseudo-code of the GPU WOAmM algorithm.}
\end{algorithm}

Random numbers are a bit troublesome. We will use the Cuda library for random numbers and use a global seed with offsets for each thread.

When random individuals are required, we will generate a random number equal to the index size of the subpopulation to  be sampled with corrections.

### Optimizations

1. Use floats for fitness calculation.
2. The population size can be set to equal the warp size 32.
3. The local copy of the population array is a trick to avoid bank conflicts, which could be feasible since the population size is small. The modulo operation will not slow the execution since we use a constant value for population size. (The butterfly reduction with __shfl_xor_sync() could be even faster since it works between registers.)
4. Within the device code, avoid pointer aliasing.
5. We can reduce the if1 and if3 predicate to a min operation to avoid warp divergence.
6. Use implicit synchronization within warp instead of the sync operation.
7. Could dynamic kernel initiation help for two or three threads?

### Improving Exploration

Few strategies to improve the stochasticity of the population, and thereby, the exploration phase follows.

1. Loop through each component OA within the while loop. Two iterations could probably give us results similar to the sequential algorithm.
2. Run multiple blocks and take the best solution over from them.

I prefer the second option to the first.

## References
