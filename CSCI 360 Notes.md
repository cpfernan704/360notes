# Lecture 1: Logistics, Intelligence, AI
* AI refers to synthetic intelligence in machines and is a field of research in computer science which aims to create it, through "The study and design of intelligent agents" or "rational agents", where an intelligent agent is a system that perceives its environment and takes actions which maximize its chances of success.
* Symbol-System Hypothesis by Newell and Simon: A physical symbol system has the necessary and sufficient means for general intelligent action.
	* reasoning = symbol manipulation
* Church Turing Hypothesis: any type of computation can be carried out on a turing machine
	* reasoning can be computational/digitized
*  4 schools of AI - thinking rationally/humanly, acting rationally/humanly

# Lecture 2: More lesson 1 stuff
* Turing Test: a computer could be said to "think" if a human interrogator could not tell it apart, through conversation, from a human being.
* Chinese Room Argument (john searle): intelligent or human-like behavior does not imply having a mind or understanding or consciousness.
* Total Turing Test: in addition to testing NLP, knowledge representation, automated reasoning, machine learning in the normal turing test, it should also include computer vision and robotics
* Descartes' dualism: mind is different from matter
	* intelligence can't be materialized
* Gödel's Incompleteness Theorem: every non-trivial (interesting) formal system is either incomplete or inconsistent
* Russell's Paradox set: $R = \{x \mid x \notin x \}$
	* the question is does $R \in R$? If it is, then it's not and vice versa
* John Von Neumann - brain can be viewed as a computing machine
	* intelligence is achievable via computation

# Lecture 3: End of Lesson 1 
*  Fuzzy sets: human systems don't deal with rigid classes. Someone who's tall might be 6'0 or 5'9 depending on who you ask.
* Neats: rigorous mathematical analysis needed to build AI
* Scruffies: we don't need rigorous mathematical analysis to build AI
* AGI - hypothetical intelligence of a machine that has the capacity to understand or learn any intellectual task that a human being can
* AGI tests: Turing, Coffee (Wozniak), Robot College Student (Goertzel), Employment (Nilsson)

# Lecture 4: Lesson 2 and start of Lesson 3
* Agents = entities that perceive their environments through _sensors_ and act upon the environment through their _actuators_
	* include humans, robots, softbots, thermostats, etc.
* Percept = the content an agent's sensors are perceiving
* Agent function : $f: P^* \rightarrow A$ maps percept histories to actions
* Agent programs implement (some) agent functions
* Rational agent chooses whichever action maximizes the expected value of the performance measure given the percept sequence to date
* Specify task environment through PEAS: Performance measure, Environment, Actuators, Sensors
* Environments can be categorized: observable, deterministic, episodic, static, discrete, single-agent
* four types of agents in increasing generality: simple reflex agents, state reflex agents, goal-based agents, utility-based agents
* Also learning-based agents which seem to be the most advanced
* Problems: Given a set of states X which constitute the problem space: Initial State $(X_I \subseteq X)$, set of goal states $(X_g \subseteq X)$, set of operators with applicability conditions $S:X' \rightarrow X, X' \subseteq X$
* Problem solving: transform the initial state into the goal state by applying a minimal sequence of operators
* ## Problem types
* Deterministic, fully observable environment means a single state problem, so solution will be a sequence and the agent will know where it is
* Non-observable environment will mean a conformant problem, so agent might not know where it is, and the solution (if any) will be a sequence
* Nondeterministic and/or partially observable means a contingency problem
	* Percepts provide new info about the current state
	* solution is a contingent plan or a policy
	* often interleave search, execution
* unknown state space means an exploration problem

# Lecture 5: Lesson 3
* A problem is defined by four items: 
	* 1. Initial state
	* 2. Successor function S(x) = set of action state pairs
		* roads: go to adjacent city with cost = distance
	* 3. goal test
		* explicit: x="at Bucharest"
		* implicit: NoDirt(x)
	* 4. path cost (additive)
		* e.g. sum of distances, number o factions executed, etc.
		* $c(x,a,y)$ is the step cost, assumed to be $\geq 0$
* A solution is a sequence of actions leading from the initial state to a goal state
* State space is a graph (X,A):
	* X: the set of nodes (all states of a problem)
	* $A \subseteq X \times X$: the set of all arcs (edges), with $(x,x')\in A$ if and only if state x' can be reached from state x by applying an operator $s \in S$
* Many AI problems can be abstracted into the problem of finding a path in a directed graph
* ## search graphs and trees
* State space graph: a mathematical representation of a search problem
	* state-nodes are abstracted world configurations
	* arcs (edges) represent successors (action results)
	* the goal test is a set of goal nodes (maybe only one)
* In a search graph, __each state occurs only once__!
* We can rarely build this full graph in memory (it's too big)
* You can convert search graphs to trees since we trees are a little bit easier to visualize and traverse
* cycle in a search graph makes a search tree infinite
* *Each node in a search tree is a path in the search graph*
* Generic search algorithm: given a graph, start state-nodes, and goal state-nodes, incrementally explore paths from the start state-nodes
	* maintain a frontier or fringe of paths from the start state-node that have been explored.
	* as the search proceeds, the frontier expands into the unexplored state-nodes until a goal state-node is encountered
	* the way in which the frontier is expanded is what differentiates the search algorithm
	* the neighbors define the graph, the goal defines what a solution is
* easier to understand a search strategy on a tree than graph, but there's no benefit for one to another in terms of implementation though
* state node is a representation of a physical configuration
* tree node is a data structure constituting part of a search tree
* ## search strategies
* a strategy is defined by picking the order of tree-node expansion
* ways of evaluating strategies:
	* completeness: does it always find a solution if one exists?
	* time complexity: number of nodes generated/expanded
	* space complexity: maximum number of nodes in memory
	* optimality: does it always find a least-cost solution?
* time and space complexity are measured in terms of:
	* b: branching factor of the search tree (max number of successors of any node)
	* d: depth of the least-cost solution (shallowest goal node)
	* m: maximum depth of the state space = depth of the search tree (may be $\infty$)
* Uninformed strategies use only the information available in the problem definition (bfs, uniform-cost search/dijkstra, dfs, depth-limited search, iterative deepening search)
* ### BFS: fringe (frontier) is a FIFO queue, I.e., new successors go at end
* properties: 
	*  complete: yes (if b is finite)
	* time: $1 + b + b^2 + b^3 + \dots + b^d = O(b^d)$
	* space: $O(b^d)$ (keeps every node in memory)
	* optimal: yes (if cost = 1 per step) not optimal in general though
	* space: is a big problem; can easily generate nodes at 100MB/sec so 24 hrs = 8640 GB
* ### DFS: fringe (frontier) = LIFO queue (aka stack), I.e. put successors at front
* properties:
	* complete: no: fails in infinite-depth spaces, spaces with loops
		* modify to avoid repeated states along path, so complete in finite spaces
	* time: $O(b^m)$; terrible if if m is much larger than d
		* but if solutions are dense, may be much faster than breadth-first
	* space: O(bm), I.e. linear space
		* _This is its major advantage_
	* optimal: no
* ### Uniform cost search (Dijkstra): fringe(frontier) = queue ordered by path cost, lowest first (priority queue)
	* This is just bfs if all the step costs are equal
* properties:
	* complete: Yes if step cost $\geq \epsilon > 0$
	* time: # of nodes with $g \leq$ cost of optimal solution, $O(b^{\lceil C^* / \epsilon \rceil})$ 
		* where $C^*$ is the cost of the optimal solution
	* space: # of nodes with $g \leq$ cost of optimal solution, $O(b^{\lceil C^* / \epsilon \rceil})$ 
	* optimal: yes - nodes expanded in increasing order of $g(n)$

# Lecture 6: ending lesson 3 stuff and starting lesson 4
* ## Search strategies
* ### Depth limited search/ Iterative Deepening Search
* idea: DFS with depth limit l, I.e. nodes at depth l have no successors, this is to eliminate the problem of time complexity especially for trees with infinite depth
* idea: *DFS space advantage with BFS time* 
	* great for shallow solutions
	* great for linear space and being complete
* how to do it?
	* Run a DFS with limit 1, if no solution run DFS with limit 2, if no solution run DFS with limit 3, $\dots$
* it's redundant, but it's still worth it
* uses a queue 
* Properties:
	* complete: yes
	* Time: $(d+1)b^0 + db^1 + (d-1)b^2 + \dots + b^d = O(b^d)$
	* Space: $O(bd)$
	* Optimal: Yes, if step cost = 1
		* Can be modified to explore uniform-cost tree
* ## Best-first search and greedy search
* Now the agents have more information than previous algorithms so we can get better time complexities
* ### Best-first search: fringe (frontier) is a queue sorted in decreasing order of desirability
* when you don't know the map, you must have an evaluation function to see how far you are from the goal
* idea: use an evaluation function to get a heuristic to determine how close you are to the goal
	* estimate of "desirability"
* heuristics are optimistic
* expand most desirable unexpanded node
* Special cases: greedy search and A*
* ### greedy search
* h(n) (heuristic)
	* estimate of cost from n to the closest goal
	* different than g which is the cost so far
* expands the node that appears to be closest to the goal - doesn't care about g at all
* properties:
	* complete: no - can get stuck in loops, but yes in finite space and repeated-state checking
	* Time complexity: $O(b^m)$, but a good heuristic can give dramatic improvement
	* Space: $O(b^m)$ - keeps all nodes in memory
	* optimal: no

* ### A* search
* uniform cost orders by path cost, or backward cost g(n), greedy orders by goal proximity, or forward cost h(n), so A* combines them
* idea: avoid expanding paths that are already expensive
* evaluation function: $f(n) = g(n) + h(n)$
* g(n) = cost so far to reach n
* h(n) = estimated cost to goal from n
* f(n) = estimated total cost of path through n to goal
* if h(n) = 0, A* becomes Dijkstra, if g(n) = 0, then A* becomes greedy
* only stop when you dequeue a goal. you can't just stop when you enqueue a goal
* A* uses an admissible heuristic so $h(n) \leq h^* (n)$ where $h^*(n)$ is the true cost from n
	* also require $h(n) \geq 0$, so $h(G) = 0$ for any goal G
	* should never overestimate the actual road distance but always exactly estimate it or underestimate it
* Theorem: A* is optimal if h is admissible
* these are the steps: enqueue, evaluate, expand, dequeue, goal test
* NB: you're only done when you dequeue goal

  ## Admissible heuristics
* for the 8 tile puzzle, you could do a heuristic to measure how many tiles are misplaced or one to measure the manhattan distance
* if $h_2(n) \geq h_1(n)$ for all n and both are admissible, then $h_2$ dominates $h_1$ and is better for search

# Lesson 7: more lesson 4 stuff
## Admissible heuristics
* the less optimistic a heuristic is, the faster A* becomes
* If you have heuristics, $h_1, h_2, \dots, h_n$, you can find a dominant heuristic $h = \max(h_1, h_2, \dots, h_n)$ 
* derived by relaxing the problem and then solving them
* Key point: the optimal solution cost of a relaxed problem is no greater than the optimal cost of the real problem

## A* search properties
* complete: no if you have an infinite search graph, yes if there are finite nodes with $f \leq f(G)$
* time: exponential in (relative error in $h \times$ length of solution.)
* space: keeps all nodes in memory
* optimal: yes -- cannot expand $f_{i+1}$ until $f_i$ is finished
* A* is an optimally efficient algorithm

## Local search and iterative improvement
* algorithms that have an eye on making search online
* in many optimization problems, path is irrelevant; the goal state itself is the solution
* state space = set of "complete-state" configurations; meaning that every state has all the components of a solution, but they might not be in the right place
	* find optimal configuration, e.g. TSP
	* find configuration satisfying constraints, e.g. timetable
* Local search algorithms operate by searching from a start state to neighboring states without keeping track of the paths. 
	* one can view this as online search, although it can be done offline
* greedy search can be turned into local search based on a heuristic, and can be viewed as online search

## Gradient ascent/descent (hill climbing/descending)
* like climbing everest in thick fog with amnesia (greedy local search)
* keeps track of the current state and in each iteration it moves to the neighboring state with optimal (highest/lowest) value
* terminates when it reaches an optimum (peak/trough) I.e. when no neighboring state is "better"
* problem: you get trapped in local extrema and you might not reach global extrema
* solution: random-restart, redo the hill climbing at random starting points
	* not guaranteed
* solution: random sideways moves escape from shoulders but loop on flat maxima

## Simulated Annealing
* idea local maxima by allowing some "bad" moves but gradually decrease their size and frequency
* Let us assume that the cost function to be minimized is E.
* How can we use a minimization algorithm to maximize a desirability function f?
* If you have a desirability function $f \geq 0$, to maximize, one way is to minimize $E = \frac{1}{f+\epsilon}$ or minimize $E = -f$
* instead of the best move, it performs random moves, and if it improves the situation, it is accepted.
* if it doesn't, then the algorithm accepts it with a probability less than one, which means flipping a coin with probability heads $=p$ to accept it or not
* The probability is proportional to an exponential function of how bad the move is, which is $\exp(- \Delta E/T)$ when descending
* we decrease T as the algorithm progresses
* T(k) = $c/(k+d)$ which is exploration/exploitation
	* c determines the initial temperature
	* d determines how fast we want the algorithm to "cool down"/make bad moves
* T simply makes bad moves, so large t means frequent bad moves, small t means infrequent bad moves
	* you'd want to allow a lot of bad moves at the beginning
* if you decrease T slowly enough you'll always reach the best state x* but this isn't an interesting guarantee because the probability of reaching the best solution is higher than the probability of any other state, but the probability of ending up in the best solution might still be very high

# Lecture 8: ending lesson 4 and starting lesson 5
* ## Local beam search
* Idea: keep k states instead of 1; begin with k randomly selected states; choose top k of all their successors
* searches that result in good states recruit other searches to join them
	* you pick the top k out of all searches
* problem: you could end up with all k states on the same local hill
* idea: choose k successors randomly, biased towards good ones

## Evolutionary algorithms
* based on the concept of natural selection and survival of the fittest
* searching to minimize a cost function or maximize a fitness function
* first you create a population of candidate solutions (strings) randomly
* then you select them that are suitable for producing offspring
* then you use a roulette wheel: give individuals chances of reproduction based on their fitness
* P parents are selected for recombination, usually p = 2
* recombination involves splitting the sequences and mating them to produce offspring
* one approach is l point crossover - only helps if substrings are meaningful components
	* crossover point is chosen randomly
* what if the crossovers result in infeasible solutions?
	* 1. discard them or algorithmically avoid having them in the first place
	* 2. penalize the infeasible solutions by reducing their fitness by a factor of how infeasible they are
* to help solutions escape from local optima, each member of the string representing the offspring are changed or mutated with a small probability called the mutation rate like 2%
	* you do this to avoid local extrema
* 1. Choose an encoding method for converting candidate solutions to a string/chromosome. Usually a binary string
* 2. Selection (Roulette wheel using fitness)
* 3. Recombination via crossover
* 4. Mutation
* 5. Repeat
* Elitism - adding some of the fittest parents to the new generation. This guarantees that the maximum fitness of the generations never decreases.
* When to stop the algorithm? track the average/maximum fitness function of each generation. When the average fitness does not change for a few generations that much, stop and select the fittest member of the last generation as your solution
* What is a fitness function and what is its argument? The argument of a fitness function is a candidate solution to a problem. For example, it can be a schedule, a solution to the TSP, a circuit layout, or a chessboard with n queens. A fitness function represents how fit/desirable the candidate solution is. For example, $f(x) = 1/\epsilon +\textnormal{ number of conflicting queens}$ 

## Gradient Based Algorithms
* Gradient methods compute the gradient vector to reduce f, e.g., by $x \leftarrow x + \alpha \nabla f(x)$. You can sometimes solve for $\nabla f(x) = 0$ exactly (e.g., with one city).
## Summary
* Heuristic functions estimate costs of shortest paths
* Good heuristics can dramatically reduce search cost
* Greedy best-first search expands lowest h
	* incomplete and not always optimal
* A* search expands lowest g+h
	* complete and optimal, and optimal efficient (expands minimal number of paths)
* Admissible heuristics can be derived from exact solution of relaxed problems
* Greedy Algorithms can be turned into online and stochastic algorithms like GA and SA

## Games
* Deterministic Games formalizations: set of states, players, actions, transition function, terminal test, terminal utilities
* Solution is a policy that assigns an action to a state
* Online search you decide your actions at each step, but in offline search you devise a solution and then go with it

# Lecture 9: lesson 5 stuff
* In zero sum games, agents have opposite objectives/utilities, so one can consider a single utility that one maximizes and the other minimizes
	* known as adversarial game playing I.e. competition only

## Adversarial search
* each agent tries to maximize their utility
* each agent has a model of its opponent and assumes that they take the "best" action
* Game playing is modeled as a search over a tree
* each node is a state and its children are states that result from each possible action at that state
* In adversarial games, at each layer, one of the agents (adversaries) takes an action
* value of a state: the best achievable outcome (utility) from that state
* the value of terminal states is known, but other states must be calculated recursively
* tic tac toe is always a draw when you play against a perfect player

##  Minimax
* original formulation is for deterministic, zero-sum games:
	* tic-tac-toe, chess, checkers
	* one player maximizes utility (us), the other minimizes utility (them, adversary)
* minimax search:
	* a state-space search tree
	* players alternate turns
	* compute each node's minimax value: the best achievable utility against a rational (optimal) adversary
		* takes the "BEST" action
* you can implement it with either double recursion or a dispatch function
* it is possible that there are more than one optimal paths
* properties:
	* resembles DFS
	* complete if tree is finite
	* optimal against an optimal opponent.
	* time complexity: $O(b^m)$
	* space complexity: $O(bm)$
	* for a game of chess, $b \approx 35, m \approx 100$
	* searching the whole game tree is infeasible

## Playing against a suboptimal adversary
* minimax analyzes worst cases
* If the probability of a mistake by the minimizer is high enough, it might make sense to take a bad path
* If min makes mistakes, we need the probability of the min making a mistake to determine the "average utility"/"expected utility"
* The expected/average utility should be bigger than what we normally get in order to take the path

## Realistic games
* problem: in realistic games like chess, we cannot search to leaves!
* solution: depth-limited search
	* instead, search only to a limited depth in the tree
	* replace terminal utilities with an evaluation function for non-terminal positions
	* evaluation function acts like a heuristic for terminal utilities (no admissibility requirements)
* guarantee of optimal play is gone
* more plies makes a BIG difference
* use iterative deepening for an anytime algorithm
* depth matters:
	* evaluation functions are always imperfect
	* the deeper in the tree the evaluation function is buried, the less the quality of the evaluation function matters
	* an important
	* example of the tradeoff between complexity of features and complexity of computation
* Evaluation functions score non-terminal states in depth-limited search
* ideal function: returns the actual minimax value of the position
* in practice: typically weighted linear sum of features:
	* $Eval(s) = w_1f_1(s) + w_2f_2(s) + \dots + w_nf_n(s)$

# Lesson 10: end of lesson 5 and lesson 6
* properties of alpha beta pruning
	* pruning has no effect on minimax value computed for the root. it is only a fourish
	* good move ordering improves effectiveness of pruning
	* with perfect ordering:
	* time complexity drops to $O(b^{m/2})$
	* doubles solvable depth
	* full search of chess is still hopeless ($35^{50}$)
	* this is a simple example of metareasoning (computing about what to compute)
* behavior is preserved under any monotonic transformation of Eval
* only the order matters: payoff in deterministic games acts as an ordinal utility function
	* you can change the numbers to whatever you want as long as you still have them in the same order and it'll still work out to the same solution

## Probability
* uncertainty refers to epistemic situations involving imperfect or unknown information
* it applies to predictions of future events, to physical measurements that are already made, or to the unknown
* Uncertainty arises in partially observable and/or stochastic environments, as well as due to ignorance, indolence, or both
* many types of uncertainty can be enumerated, from different viewpoints:
	* vagueness/fuzziness: the difficulty of making sharp or precise boundaries for concepts in the world. Example: tall
	* ambiguity: associated with one-to-many relations, I.e., situations with two or more alternatives that are left unspecified. Example: dice with interval payoffs
	* probabilistic: related to chance and randomness. Is it real?
* general situation in AI:
	* observed variables (evidence): agent knows certain things about the state of the world (e.g. sensor readings)
	* unobserved variables: agent needs to reason about other aspects (e.g. where an object is or what disease is present)
	* model: agent knows something about how the known variables relate to the unknown variables
* probabilistic reasoning gives us a framework for managing and combining our beliefs, knowledge, and measurements

# Lecture 11: more lesson 6 stuff
* Conditional probability: $P(A\mid B) = \frac{P(A \cap B)}{P(B)}$
	* $P(A \cap B) = P(A \mid B) P(B) = P(B \mid A) P(A)$
* Independence: $P(A \cap B) = P(A)P(B)$
* Independence of two events A and B does not imply that they are mutually exclusive/disjoint/non-overlapping
* Mutually exclusive events A,B: $A \cap B = \emptyset$
* The empty set is mutually exclusive with any event and is independent from any event
* If at least one of $P(A)$ or $P(B)$ are zero, then they are independent
* Bayes' Theorem: $P(A\mid B) = \frac{P(B \mid A) P(A)}{P(B)} = \frac{P(B \mid A) P(A)}{P(B \mid A) P(A) + P(B \mid A^c)P(A^c)}$
	* $P(A|B)$ = posterior
	* $P(B\mid A)=$ likelihood
	* $P(A)$ and $P(B)$ are priors
	* posterior $\alpha$ likelihood $\times$ prior

# Lesson 12: more lesson 6 stuff
* PDF of a normal random variable: $f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/2\sigma^2}$ 
* $E[X] = \sum_{m} x_m P(X=x_m)$
* $Var[X] = \sum_{m} (x_m-E[X])^2 p_x(X=x_m)$
	* $Var[X] = E[X^2] - (E[X])^2$
* What does the variance mean? The average/expectation of a random variable is not a good measure of its dispersion.
	* it magnifies how much a random variable fluctuates around the mean

# Lesson 13: final lesson 6 stuff and lesson 7 stuff 2/26
* We say X and Y are independent random variables if $p_{XY} = P(X=x \cap Y = y) = P(X=x) P(Y=y) = p_x (x) p_y (y)$
* goal of ML: build agents that adapt to new situations and learn, I.e. change their internal beliefs and structures, using their experience, which is usually translated into data
* the subfield of AI that focuses on designing agents that learn from data is called Machine Learning or Statistical Learning
* types of learning:
	* supervised learning: goal: prediction
	* unsupervised learning: goal: discovery of similarity or dissimilarity, goal: dimensionality reduction
	* reinforcement learning: categorized as supervised learning as well
* Supervised learning
	* starting point: you have outcome measurement Y (target) and predictor measurements X (features), so you want to predict/estimate Y based on X
	* if Y is quantitative, then you have a regression problem
	* if Y is qualitative, then it's a classification problem
	* on training data, we want to accurately predict unseen test cases, so generalize
		* also understand which inputs affect the outcome, and how
		* also assess how good our predictions are
* Unsupervised learning
	* no outcome variable, just a set of predictors (features)
	* objective is vague - find groups of samples that behave similarly
	* difficult to know how well you're doing, but can still be useful as a pre-processing step for supervised learning

# Lecture 14: lesson 7 stuff 2/28
* Reinforcement Learning
	* Setting: learning goal is to maximize rewards
	* Used to learn models of how to behave, more complex than just input-output
	* used for designing autonomous robots and automated vehicles
	* can also be trained to play games
* Classifiers
	* Is there an ideal optimal classifier? Yes, if you have K elements in C, numbered 1 through K, then $p_k(x) = P(Y=k \mid X=x)),k = 1,2,\dots,K$ 
	* These are conditional class probabilities at x; e.g. see little barplot at x = 6. Then the Bayes optimal classifier at x is $C(x) = j$ if $p_j(x) = \max\{p_1(x), p_2(x), \dots, p_K(x)\}$
		* classify to the class that gives you the largest class conditional probablility
	* Big issue: if you have insurance claims, only like $3\%$ of people have a claim. Even though that's a negligible minority, you can't just ignore them since the goal of insurance companies is to cater to those $3\%$ 
	* Majority polling = find the class that holds the majority in a neighborhood
* k-Nearest Neighbors
	* basic idea: if it walks like a duck, quacks like a duck, then it's probably a duck
	* requires 3 things
		* set of labeled records
		* distance metric to compute distance between records
		* the value of k, the number of nearest neighbors to identify
	* determine the neighborhood by moving away from your test record so far to find the k nearest neighbors
	* can also fix the distance we want to move away from the test point instead of k to find k nearest neighbors




# Lecture 15: lesson 7 stuff 3/4
* For kNN, to identify a test point, compute distance to other training records, identify the k nearest neighbors, then use class labels of nearest neighbors to determine the class label of unknown record (take the majority)
* Evaluating classifiers
	* we measure the performance of our classifier $\hat{C}(x)$ using the misclassification error rate, which is the fraction of data points that were not classified correctly
	* Bayes classifier has smallest error
* Noise can change the class of data points. Sometimes there will be negative in the positive and positive in the negative
* the lower the k, the more noise sensitive it is, so it will overfit
* the higher the k, the less noise sensitive it is, so it will underfit and add points from other classes
* we choose the best value of k by evaluating a classifier on unseen data. If it can generalize to new data it's good
* for different datasets, different distance measures might be appropriate
	* $\mathscr{l}_1$ norm is the manhattan distance between two vectors
	* $\mathscr{l}_p$ norm can be calculated by: $(|x_1 - x_1^{*}|^p + \dots + |x_p - x_p^{*}|^p)^{\frac{1}{p}}$
	* so, we want to find the best $T(k,p)$ where k can be anything from 1-100 and p can be anything from 1-100. Choose the (k,p) that minimizes T
	* parameters that are selected by examining a test set are called hyperparameters
* Data preprocessing
	* sometimes features may have to be scaled so that distance measures aren't dominated by one of the features
	* might not always help - if income is the only relevant feature, it is actually beneficial that income dominates the calculation of the distance measure. Scaling here might not help/worsen the ML algorithm
* Irrelevant feature - feature that provides the agent little discrimination power among class. You want features with high discrimination power
	* E.g. $P(IQ=105 \mid Y = \textnormal{male}) = P(IQ = 105 \mid Y = \textnormal{female})$
* Redundant feature - doesn't provide much additional information to the agent about classification. Non-redundant features might provide the agent high discrimination power
* kNN is sensitive to redundant and irrelevant features
	* irrelevant features confuse the distance measure
	* redundant features bias the distance measure towards certain features
* kNN advantages
	* simple to implement
	* few tuning parameters (K, distance metric)
	* flexible, classes don't have to be linearly separable
* kNN disadvantages
	* computationally expensive - must compute distance from new observation to all known samples
		* also have to sort distances
	* sensitive to imbalanced datasets - may get poor results for infrequent classes
	* sensitive to irrelevant and redundant inputs - makes distances less meaningful for identifying similar neighbors
* Regression
	* we model the relationship between the predictors and the dependent variable as $Y = f(X) + \epsilon$ where X is the vector of p features
	* we need $\epsilon because Y is not usually fully explained by $X_j$
	* Conditional expectation is the ideal or optimal predictor of Y with regard to mean-squared prediction error
		* if we choose mean absolute error $E[|Y-g(x)| \mid X = x]$ as our criterion, then the median of $Y \mid X = x$ will be the best estimate. Useful when we have a lot of outliers
	* nearest neighbors regression: $\hat{f}(x) = \textnormal{average}(Y \mid X \in N(x))$
		* we choose a neighborhood of x that has k closest samples in it
		* we need to choose a distance measure and k (or a similarity measure that specifies what "close" means)

# Lecture 15: end of lesson 7 and start of lesson 8 3/6
* you can't just average errors because positive and negative errors cancel out, so we average squared errors
* average of 1,2,3 =median of 1,2,3
* average of 1,2,1000 is not median of 1,2,1000
	* median isn't as susceptible to outliers
* Assessing accuracy of regression models
	* we could compute the average squared prediction error on the training set: mean squared error
		* $\frac{1}{n} \sum_{x_i \in Tr} [y_i - \hat{f}(x_i)]^2$
		* an error of 0 might not be good since it could just be overfit
		* favors overfit models
* if k is too small, the model fails to generalize, but if it's too big, it doesn't learn from the data at all
	* choose k that has the minimum MSE on a test set
* Classification boils down to calculating the posterior probability $P(Y=k \mid X=x)$ 
	* we can approximate with kNN, but we can also calculate it with Bayes' Rule
* From Bayes' Rule:
	* $P(Y = k \mid X = x) = \frac{P(X=x \mid Y=k)P(Y=k)}{P(X=x)}$
	* $P(Y=k \mid X = x)\textnormal{ } \alpha \textnormal{ }P(X = x \mid Y = k) P(Y =k)$
* Usually we don't have the population information
* Instead, we should estimate the prior probabilities of each class and the likelihoods from data
* Estimating the prior $P(Y=k)$ is a matter of calculating the proportion of data that is from each class
* Estimating $P(X=x \mid Y = k)$, if discrete you can build tables, if continuous then discritize them
* To estimate the joint distribution of features using histograms, we need exponentially growing data $M^p$
* This is called the curse of dimensionality
	* example: we need many women in our dataset that have short hair, wear blue, have above average IQ, are college educated, whose eye colors are brown and hair colors are brown, if those are our features
* Huge assumption we can make to help the curse of dimensionality: Naive Bayes Assumption)
	* we assume that features are independent in each class
	* $P(X = x \mid Y = k) = P(X_1 = x_1, X_2 = x_2 \dots X_p = x_p \mid Y = k)$
	* $= P(X_1 = x_1 \mid Y = k) P(X_2 = x_2 \mid Y = k) \dots P(X_p = x_p \mid Y=k)$
	* Things can be not independent but be conditionally independent
		* $P(Shoe size = x, reading level = y) \neq P(Shoe size = x) P(reading level = y)$
		* $P(Shoe size = x, reading level = y \mid Age = 6)$
		* $=P(Shoesize = x \mid age = 6)P(readingsize = y \mid age = 6)$
	* with this assumption, we can estimate the probability distribution of each feature in each class, regardless of how features interact

# Lecture 16: lesson 8 stuff 3/18
* Naive Bayes gives you incorrect probabilities but correct decision
* to deal with continuous features, discritize the data so that it falls into bins
* You could also replace the probability $P(X_r = x_r \mid Y = k)$, with the probability density function of X in each class I.e. $f_{x_r \mid Y}(x_r \mid k)$
	* you can show that Bayes rules is still correct when replacing probabilities with pdfs
	* any type of pdf can be used, but Gaussians are popular
		* $f_{x_r \mid Y}(x_r \mid k) = \frac{1}{\sigma_k \sqrt{2\pi}} e^{- \frac{-(x-\mu_k)^2}{2\sigma_k^2}}$
		* $\mu_k$ is the mean of the gaussian feature, $\sigma_k$ is the standard deviation of the Gaussian feature
* If you don't know $\mu_k, \sigma_k$
	* estimate: $\hat{\mu}_k = \frac{1}{n_k} \sum_{i \mid y_i = k} x_i$
	* estimate: $\hat{\sigma}_k = \frac{1}{n_k - 1} \sum_{i \mid y_i = k} (x_i - \hat{\mu}_k)^2$
* Sometimes if there's a lack of data, Naive Bayes might accidentally say the probability of something happening is 0 just because there's no observations of that thing happening
	* remedy: Laplace smoothing
	* $P(X_j = c \mid y) = \frac{n_c + 1}{n + v}$ where v is the number of options that feature can have
* When dealing with real world data, especially when the number of features is large, it is a general recommendation to "turn on" the Laplace Smoothing method
* Naive Bayes properties
	* robust to limited number of noisy data
	* handling missing values by ignoring the missing feature during probability estimate calculations
	* robust to irrelevant attributes
	* redundant and correlated features violate the conditional independence assumptions, so not robust to redundant features

# Lecture 17: lesson 8 stuff 3/25
* Redundant features are features that carry the same amount of information, so they are correlated. They violate the independence assumption
* Instead of global independence we can assume "local independence"
	* instead of separating each feature, keep correlated features grouped up
* Bayesian Belief Network
	* The structure of the joint probability distribution can be learned from data, but it is beyond the scope of this course
	* total conditional independence might be too strong an assumption for data, so we appeal to weaker assumptions that can be loosely called "local conditional independence"
	* local conditional independence in a probability distribution can be represented by directed acyclic graphs called bayesian Belief Networks
	* each node corresponds to a variable (feature or label)
	* arcs represent a dependence relationship between variables
	* a probability distribution is associating each node to its "immediate parent"
	* a node in a Bayesian network is conditionally independent of its non-descendants, given its parents are known)
	* Naive Bayes is a very specific form of a Bayesian Network
	* If X has only one parent (Y), table contains conditional probability $P(X \mid Y)$
	* If X has multiple parents, the table contains conditional probability $P(X \mid Y_1, Y_2 \dots Y_k)$
	* Order doesn't matter when you write the equation
	* PRACTICE THIS FOR EXAM

# Lecture 18: lesson 8 stuff and lesson 9 stuff 3/27
* no algorithm is the best for ML. Each will perform well on some task/dataset. None isn't the best for everything
* No Free Lunch Theorem: all algorithms perform equally when averaged over all possible problems
	* generalization can only be obtained by assumptions
* Assume that conditional expectation is a linear function
* In simple linear regression, we assume a model $Y = \beta_0 + \beta_1 X + \epsilon$ where $\beta_0$ is the intercept, $\beta_1$ is the slope, and $\epsilon$ is the error term
	* we estimate these parameters from data
* $\hat{\beta}_0$ and $\hat{\beta}_1$ are estimates where we can predict future labels using $\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$
	* $e_I = y_I = \hat{y_I}$ represents the $i^{th}$ residual
	* $RSS = e_1^2 + e_2^2 + \dots + e_n^2$
	* $RSS = (y_1 - \hat{\beta_0} - \hat{\beta_1}x_1)^2 + \dots + (y_n - \hat{\beta_0} - \hat{\beta_1}x_n)^2$
* Residual sum of squares can be viewed as a loss function
	* why do we square the values? if we have positive and negative errors, they are equally bad, but they might cancel and we'll get a good loss function when that might not be the case
* This is supervised learning where we know all $x_i, y_i$, but we don't know the $\beta_0$, $\beta_1$
* To minimize L, we want to minimize $L(\hat{\beta_0}, \hat{\beta_1})$
	* take the gradient, set it to 0, then you get these solutions:
	* $\hat{\beta_1} = \frac{n \sum_{i=1}^n x_iy_i - (\sum_{i=1}^n x_i)(\sum_{i=1}^n y_i)}{n\sum_{j=1}^n x_j^2 - (\sum_{j=1}^n x_j)^2}$
	* $\hat{\beta}_0 = \frac{1}{n}(\sum_{j=1}^n y_j - \hat{\beta_1} \sum_{j=1}^n x_j)$
* Multiple Linear Regression
	* our model is $Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon$
	* $\beta_j$ as the average effect on Y of a one unit increase in $x_j$, holding all other predictors fixed
* The ideal scenario is when the predictors are uncorrelated
	* each coefficient can be estimated and tested separately
* correlations amongst predictors cause problems:
	* interpretations becomes hazardous - when $X_j$ changes, everything else changes
	* claims of causality should be avoided for observational data
* Example
	* Y total amount of change in your pockets
	* $X_1 = \#$ of coins
	* $X_2 = \#$ of pennies, nickels and dimes
	* By itself, regression coefficient of Y on $X_2$ will be positive. But, how about with $X_1$ in model?
		* $Y \approx 3X_2$, and if $X_1 = 2X_2$, then $Y = 2X_1 - X_2$
			* it's ok to say that one of the coefficients is negative
			* you can't just look at the $\beta_2 = -1$ and say you don't have a lot of pennies nickels and dimes
* Least squares for multiple regression
	* we can make predictions using the formula: $\hat{y} = \hat{\beta}_0 + \hat{\beta_1}x_1 + \dots + \hat{\beta_p}x_p$
	* $RSS = \sum_{i=1}^n (y_i - \hat{\beta_0} - \hat{\beta}_1x_{i1} - \dots - \hat{\beta_p}x_{ip})^2$
	* we don't know the $\hat{\beta}$ so we estimate them with software
* We can also do polynomial regression: $Y = \beta_0 + \beta_1 X + \beta_2 X^2$

# Lecture 19: lesson 10 stuff 4/1
* tree based methods - segmenting the feature space so that classification and regression are easier
* pros: simple to implement, easy to understand/interpret
* con: not competitive with other methods in terms of accuracy
## Regression trees
* decision trees discritize
* broken into terminal and internal nodes.
* oversimplification, but easy to display, interpret, and explain
* predictor space needs to be split into distinct and non-overlapping regions
* for every observation that falls into a region, we make the same prediction
	* prediction = mean of response values for training observations in that region
* algorithmically burdensome to split data into regions since there's so many possibilities. $2^n$ options since for each data point, you have to decide whether it goes into a region or not
* since we can't exhaust all possible partitions to find the smallest RSS, we'll use a top down greedy to find the best option, then the next best option
	* doesn't guarantee optimality for the future - you might not find the best RSS with this
* first split should lead to the greatest possible reduction in RSS
	* rinse and repeat, but you only split one of the regions you split in the last step
	* reduction in RSS = gain
* stop when no region has more than some amount of observations like five
* you don't want every region to just have one observation, otherwise it's just going to be a lookup table/be overfit
* two problems or issues
	* 1 A piece-wise constant model 
		* use a model in each region, for example KNN or linear regression. The resulting decision tree is called a model tree
	* 2 Discontinuity
		* blur the region boundaries - a data point can be a member of more than one regions with different degrees of membership
			* prediction associated with it is an aggregation of the predictions provided by each of the regions based on the degree of membership in each region
			* "Fuzzy sets"
* regions with a lot of observations should have more power
	* use a weighted residual sum of squares
		* RSS for a region times number of members in that region
	* more popular for classification trees but still used for both classification and regression trees
* in classification we're predicting a qualitative variable, in regression we're prediction a qualitative variable

## Classification trees
* in each region,  we predict the most commonly occurring class as the label of that region - majority vote
* want to minimize misclassification, each region must be very homogenous
* classification error rate is the loss function
	* fraction of the observations for a region that have been misclassified
	* error = 1 - $\max p_{mk}$ 
* classification error is not sufficient enough for tree-growing, so two other measures are preferable
* Gini index = $$G = \sum_{k=1}^K \hat{p}_{mk}(1 - \hat{p}_{mk})$$
	* calculate it for each region
	* sum of variance for each class
	* small when regions are close to homogenous - pmk should be either close to either 1 or 0
	* measure of purity 
* Cross entropy/Deviance $$D = - \sum_{k = 1}^K \hat{p}_{mk} \log{\hat{p}_{mk}}$$
	* for pure regions, we still have a small D
* In classification, we usually like weighted Gini or cross-entropy more.
* find the weighted purity measure (P) before splitting, weighted purity measure (M) after splitting, and then choose the split that gives us the highest gain $P-M$
* Finding the best split
	* split into two partitions
	* effect of weighing partitions
	* larger and purer partitions are sought
* to deal with overfitting, it's better to be underly cautious than overly cautious
	* If we're overly cautious, we might not reach a good split and terminate too early once we dont get much change in splits
	* if we're underly cautious, we can build a large tree that's overfit then prune it back

# April 3
* To avoid the decision tree from overfitting, we can try to stop training sooner - this is bad since we might stop at a bat split even though it would give rise to better splits in the future
* Other strategy - growing a tree to be really large then pruning
* NOT RESPONSIBLE FOR EXAM:
* We first build a large decision tree $T_0$, then we index its subtrees by a parameter $\alpha$ in the following way:
	* find a sequence of parameters, and for each parameter in that sequence/alpha, you find the subtree that has the lowest $J$
		* $J = \textnormal{Loss} + \alpha |T|$
			* Loss can be RSS for regression, Gini for classification
		* when alpha is really small, we select the largest tree, $T_0$ itself
			* the larger the tree, the better the classification/regression
		* when alpha is really big, we select the smallest subtree (no-split $|T| = 0$) 
		* the smaller the alpha, the more overfit the subtree is with a lot of splits
		* when alpha is really large, the subtree is really underfit and not split much (null model)
		* we can control how fit we want our tree
		* larger models have better loss but are prone to overfitting, so $\alpha |T|$ is a penalty
		* alpha controls a trade off between complexity of subtree and its fit to the training data
		* find the best alpha by using a test set - find what alpha gives you the best loss/misclassification, then return to the full data set and obtain the subtree corresponding to the best alpha
* All algorithms in machine learning are to some extent black boxes
* Models are data-driven - configured from the data. This leads us to problems such as
	* how should we interpret the models
	* how to ensure that models are transparent in their decision making
	* how do we make the results of these algorithms are fair and statistical fairness
* Issue of trust
	* cooperation between agents, in this case algorithms and humans, depends on trust
	* if humans are gonna accept algorithmic prescriptions, they need to trust them
	* incompleteness in formalization of trust criteria is a barrier to straightforward optimization approaches
	* for that reason, interpretability and explainability are posited as intermediate goals for checking other criteria
	* interpretability doesn't have a mathematical definition either
	* sometimes AI learns undesirable tricks to meet its goals
* 2 approaches for trust
	* model specific techniques: only suitable for use by a single typeof model. For example, layer visualization is only applicable to neural networks
	* model agnostic: they can be used as wrapper techniques with any model
	* example: decision trees trained on other models to explain their output
	* model agnostic techniques generally involve examining the input or output data distribution
* Look at this lecture after the break for exam question

# Lecture 4/10
* REWATCH FIRST PART
## Neural networks
### Perceptron
* Perceptron is an algorithm for binary classification that uses a linear prediction function
* $\beta^T(x) + \beta_0 = \beta_0 + \beta_1(x_1) + \beta_2(x_2) + \dots + \beta_n x_n$
* if this expression is nonnegative, evaluate to 1, otherwise evaluate to -1
* also called the sign function
* $\beta^T$ called the weights
* $\beta_0$ called the bias
* ties are broken in favor of the positive class by convention
* perceptrons learn the weights from the data
* learns the weights iteratively by first initializing them all to 0 or randomly
* then present the training data one at a time and classify the instant. If the prediction is correct, don't do anything, if it's incorrect, update it using an update rule
* rinse and repeat 
* Example of an update rule:
	* if you got $\beta^Tx + \beta_0 < 0$ and you want $\beta^Tx + \beta_0 \geq 0$, then adjust the weights $\beta$ so that the function becomes positive or moves towards being positive, and vice versa
