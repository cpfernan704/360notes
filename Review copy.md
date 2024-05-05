
## Problem 3
$f, g, \in C^0 (\mathbb{R})$
a) Prove $\bar{\mathbb{Q}} = \mathbb{R}$
First, $\bar{\mathbb{Q}} \subseteq \mathbb{R}$ is trivial. For $\supseteq$, given an arbitrary $x \in \mathbb{R}$, WLOG, $x \notin \mathbb{Q}$, We want to show $\forall \epsilon > 0$, $\exists q \in \mathbb{Q}$. $|q - x| < \epsilon$. $\epsilon > 10^{-n}$ for some $n \in \mathbb{N}$. $|\frac{[10^n x]}{10^n} - x| < 10^{-n} < \epsilon$.
$\implies x$ is a limit point of $\mathbb{Q}$
For an alternative solution, look at Zorich - it uses the Archimedean principle

b) if $f \mid_\mathbb{Q} = g\mid_\mathbb{Q} \iff f = g$
Proof: $f =g \implies f \mid_\mathbb{Q} = g\mid_\mathbb{Q}$ is trivial by definition of restriction. For the other direction, we want to prove $\forall x \notin \mathbb{Q}$, $f(x) = g(x)$. By a, we know that x is a limit point of $\mathbb{Q}$, $\therefore$ $\lim_{\mathbb{Q} \ni q \rightarrow x} f\mid_\mathbb{Q}(y) = f(x)$. This is true because we know f is continuous.  Then, since $g\mid_\mathbb{Q} = f\mid_\mathbb{Q}$, $\lim_{\mathbb{Q} \ni y \rightarrow x} f\mid_\mathbb{Q}(y) = \lim_{\mathbb{Q} \ni x} g\mid_\mathbb{Q} (y) = g(x)$ 
QED

c) Prove that $|C^o (\mathbb{R})| =|\mathbb{R}|$ 
Proof: We'll use Schroder Bernstein. $\mathbb{R} \leq |C^o (\mathbb{R})|$: $\phi(x) = f_x$ where $f_x: \mathbb{R} \rightarrow \mathbb{R}, f_x (y) = x \forall y \in \mathbb{R}$ 
For the other direction $|C^o (\mathbb{R})| \leq \mathbb{R}$ , $\phi(x)=f_x$. By part b, $|C^o (\mathbb{R})| =|C^o (\mathbb{Q})| \leq |\mathbb{R}^\mathbb{Q}| = |\{0,1\}^\mathbb{N}| = |\{0,1\}^{\mathbb{N} \times \mathbb{Q}}|= |\{0,1\}^\mathbb{N}| = |\mathbb{R}|$

d) Prove that there does not exist an injection from $\{h: \mathbb{R} \rightarrow \mathbb{R}\}$ to $\mathbb{R}$ 

$|\{f: \mathbb{R} \rightarrow \mathbb{R}\}| = |\mathbb{R}^\mathbb{R}| \geq |\{0,1\}^\mathbb{R}| = |\mathscr{P}(\mathbb{R})| > |\mathbb{R}|$ where the last inequality holds by Cantor's theorem


## Problem 4
![[Screen Shot 2024-05-04 at 6.53.15 PM.png]]
Proof: Given $x\notin \mathbb{Q}$, we want to show that $\forall \epsilon > 0, \exists \delta_\epsilon > 0$ such that $|x-y| < \delta_\epsilon \implies |f(y)| < \epsilon$ . $\exists N \in \mathbb{N}: \frac{1}{N} < \epsilon$ by the Archimedean property. Consider the set $A=\{a \in (0,1)\mid f(a) > \frac{1}{N}\}$ . If $a \in A:a = \frac{m}{n}$ such that $n < N$ because $f(a) = \frac{1}{ n} > \frac{1}{N}$. Note that since $a \in (0,1)$, $A = \bigcup_{n=2}^{N-1} \{\frac{1}{n} \dots , \frac{n-1}{n}\} \implies A$ is finite. Therefore, take $\delta_\epsilon = \min_{a \in A} \{|x-a|\}$. Then, if $|y-x| < \delta_\epsilon \implies y \notin A \implies |f(y)| \leq \frac{1}{N} < \epsilon$ 

## Problem 5
![[Screen Shot 2024-05-04 at 7.12.22 PM.png]]
Part a) ($\implies$) $\inf_{z \in E}d(x,z) = 0$. This means that $\forall \epsilon > 0, \exists z \in E: d(x,z) < \epsilon$. If $x \in E$, $\inf_{z \in E} d(x,z) \leq d(x,x) = 0$ $\implies x \in \bar{E}$. 
$(\impliedby)$ if $x \in \bar{E} = E \cup \{ \textnormal{limit point of } E\}$ 
$(1) x \in E \inf_{z \in E} d(x,z) \leq d(x,x) = 0$  $\implies \rho_E d(x,z) = 0 (d(x,z) \geq 0, \forall z)$ 
(2) $x \in \{ \textnormal{limit point of } E\}$ $\forall \epsilon > 0, \exists z \in E: 0 < d(x,z) < \epsilon$ $\implies \inf_{z \in E} d(x,z) = 0$

Part b) Proof: $\rho_E (x) \leq d(x,z) \forall z \in E$
$\leq d(x,y) + d(y,z)$ by the triangle inequality
Let $S=\{d(x,y) + d(y,z) \mid z \in E\}$
$\rho_E$ is a lower bound of S $\implies$ $\rho_E \leq \inf_{z \in E} d(x,y) + d(y,z)$, which can be simplified to  $\rho_E (x) \leq d(x,y) + \rho_E (y)$. Then, $\rho_E (x) - \rho_E (y) \leq d(x,y).$. For the same argument, $\rho_E(y) \leq d(y,z) \forall z\in Z$
$\leq d(y,x) + d(x,z)$
$\implies \rho_E(y) - \rho_E(x) \leq d(x,y)$
Therefore $|\rho_E(x) - \rho_E(y)| \leq d(x,y)$

## Problem 6
![[Screen Shot 2024-05-04 at 7.50.04 PM.png]]
$\rho_F$ $(x) = \inf_{z \in F} d(x,z), x \in X$, $\rho_F : X \rightarrow \mathbb{R}_{\geq 0} \rho f$ is uniformly continuous
$\min \rho_F(k) \in \rho_F(k)$  by the Extreme Value Theorem because k is compact. $\rho$ is uniformly continuous $\implies$ $\rho$ is continuous. K is compact which implies that its supremum and infimum are realized.
$B = \min \rho_F(k)$ 
(1) $\rho_F (k) = 0$ iff $k \in \bar{F} = F$ by the last problem, which is a contradiction.
$\implies B > 0$ 
Given $p \in k, q \in F$, $d(p, q) \geq \inf_{r \in F} d(p, r)$
$= \rho_F(p)$
$\geq \min \rho_f(p) = B$
$\delta = \frac{B}{2}$
