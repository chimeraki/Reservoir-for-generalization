# Reservoir-for-generalization

We investigate the ways in which a machine learning architecture known as Reservoir Computing learn concepts such as `similar' and `different', and other relationships (such as rotation, scaing, blurring) between image pairs, and generalize these concepts to previously unseen classes of data. We present two reservoir computing architectures (single and dual), that broadly resemble neural dynamics and show that a Reservoir Computer (RC) trained to identify relationships between image-pairs drawn from a subset of training classes, generalizes the learned relationships to substantially different classes unseen during training.
Additionally, we observe an attractor structure in reservoir dynamics (the high dimensional reservoir state clustered using PCA and t-SNE), i.e., the reservoir system state trajectories reach different attractors (patterns) representative of corresponding input-pair relationships. Thus, as opposed to training in the entire high dimensional reservoir space, the RC only needs to learn the attractor structure, allowing it to perform well in the task of generalization with very few training examples (few hundred) compared to conventional machine learning techniques such as deep learning. 
We also show that RCs can not only generalize linear as well as non-linear relationships, but also combinations of relationships



![Single reservoir architecture](https://github.com/chimerask/Reservoir-for-generalization/blob/master/Images/one_reserv.jpeg)
Reservoir architecture with input state of the two images at time t denoted by $\protect\vv{u}(t)$, reservoir state vector at a single time by v(t)$ and output state by y(t)$.

![Dual reservoir architecture](https://github.com/chimerask/Reservoir-for-generalization/blob/master/Images/two_reserv.jpeg)
