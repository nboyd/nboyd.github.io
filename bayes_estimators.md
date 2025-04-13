# Notes on Bayes Estimators in Machine Learning

In this note we'll review a few applications of a simple but powerful idea from statistics, the *Bayes estimator*, to machine learning, culminating in very a simple proof that the training problem for denoising diffusion probabilistic models has an *analytic solution* that cannot generalize.

XXX powerful tool to help understand ML ...

### A review of Empirical Risk Minimization
---


In the typical machine learning (theory!) setup we assume the existence of a data-generating distribution[^1] $p$ on $\mathcal{X} \times \mathcal{Y}$. Typically we only have access to a fixed-size set of IID samples from $p$: $(x_1, y_1), \ldots, (x_n, y_n)$, but, as we'll see, there are some situations in which we have direct access to $p$.

As a concrete example take *cat-classification*: the task of detecting the presence of a cat in a natural image. $\mathcal{X}$ in this case is some space of images and $\mathcal{Y}$ is the set $\{\textrm{cat}, \textrm{no cat}\}$. $p$ is the distribution given by sampling a random image $x$ from all possible images (say on the internet) and $y$ is $\textrm{cat}$ if and only if there is a cat in the image.

The goal[^2] of machine-learning is to choose *parameters* $\theta$ such that the neural network $f_\theta$ that predicts $y$ from $x$ achieves low *risk*:

$$ \mathbf{E}_{(x,y) \sim p} \ell(f_\theta(x), y). $$

Here $\ell: \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}$ is our *loss*: a number that is large if we're unhappy with the prediction $f_\theta(x)$ when the true label is $y$, and small otherwise. The risk is simply the average of the loss under the data-generating distribution $p$. 

The standard training objective is then to minimize the risk:

$$ \textrm{minimize}_\theta \mathbf{E}_{(x,y) \sim p} \ell(f_\theta(x), y).$$

In the typical ML setup where we only have a sample from $p$ we instead minimize the *empirical risk*, replacing the expectation over $p$ with the expectation over the empirical distribution:

$$ \textrm{minimize}_\theta \frac{1}{n} \sum_i \ell(f_\theta(x_i), y_i).$$

This procedure is called empirical risk minimization (ERM).

### Bayes Estimators
---


Much has been made of the fact that most neural networks are *universal function approximators*: essentially we can choose $\theta$ such that $f_\theta$ is arbitrarily close to any function $g$ on $\mathcal{X} \times \mathcal{Y}$. In typical practice this doesn't matter: it tells us nothing about how well ERM will generalize, how well we can approximate the ERM solution using SGD, etc. However, taking this idea seriously allows us to predict the behavior of ERM in a few interesting cases: on the training data, when we know $p$, and as we'll see for diffusion models.

Assuming our NN is a universal function approximator allows us to replace the minimization over $\theta$ with a minimization over all functions from $\mathcal{X}$ to $\mathcal{Y}$; when we do this we find that the risk minimization problem has an __analytic solution__:

$$ g^*(x) = \textrm{argmin}_{\hat{y} \in \mathcal{Y}} \mathbf{E}_{y \sim p (y | x)} \ell(\hat{y}, y). $$

To see why this is the case factor $p(x,y)$ into $p(x) p(y | x)$ and pull the expectation over $x$ out:

$$ \mathbf{E}_{(x,y) \sim p} \ell(g(x), y) = \mathbf{E}_{x \sim p(x)} \mathbf{E}_{y \sim p(y | x)} \ell(g(x), y). $$

Because $g(x)$ can be any $y \in \mathcal{Y}$, we are free to choose $g(x) = \hat{y}$ to minimize the inner expectation over $y$.
The resulting function $g^* : \mathcal{X} \rightarrow \mathcal{Y}$ is called a *Bayes estimator*.


### What can we do with this idea?
---


In the typical ML setup where we don't have access to $p$ but rather only a sample, this idea gives us very little insight: $g^*$ is defined only at the datapoints $x_i$ (which we don't care about) and tells us nothing about the behavior of $f_\theta$ on new data (which we *do* care about)!

That said we'll give a few examples of interesting cases where the behavior of ML models does follow the Bayes estimator and that behavior is interesting.

#### Interpolation
---

One trivial application is to show that ERM with a NN interpolates the training data: if the loss function is reasonable (e.g. squared error, cross entropy, etc) and the training data are *unique* (i.e. there is only one $y$ for each $x$) then we have 

$$ g^*(x_i) = y_i.$$

While uninteresting this does match practice. 

#### Multiple labels
---

One common situation in ML is when the training inputs $x_i$ are not unique. 
For example, in the protein structure prediction problem the input $x$ is the primary structure (sequence of amino acids) and the output $y$ is a 3D structure. Because proteins are not completely static there are often many 3D structures associated with the same amino acid sequence.

If the training input $x$ is associated with the labels $y_1, \ldots, y_k$ one reasonable question is what prediction will ERM make for $x$. A typical assumption is that it will simply pick one of the $y_i$ at random. This is not the case: the Bayes estimator will make a prediction that is an $ell$-mediod of the $y_i$:
    $$ g^*(x) = \textrm{argmin}_{y \in \mathcal{Y}}\frac{1}{k} \sum_i \ell(y, y_i). $$

For some applications this makes perfect sense; in classification with the cross entropy loss $g^*$ is the empirical posterior over class labels.

In the protein structure application, however, this makes absolutely no sense: the predicted structure is an average that is not a valid protein structure (i.e. it violates all kinds of physical constraints and could never occur in nature). 

#### Simulations with infinite data



[^1]: Often this assumption is incorrect.

[^2]: Not strictly true: often the training loss is only a proxy for the true objective.