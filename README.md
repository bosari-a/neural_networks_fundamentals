# Neural Networks from scratch (solving XOR)

XOR stands for “exclusive or.” This is an operator that takes in exactly two binary inputs and only returns true (or 1) if the inputs are different. If the inputs are the same, even if both are true/1, XOR returns false (0). The graph of XOR is not linearly separable and because of that it is not possible to model XOR using a linear relationship.

![XOR](/imgs/xor.webp)

The way a neural network tries to achieve that is by taking in the input vector (for example a possible input could be (0,1) where the solution to this is 1 (true)) and then it initializes random weights. After the first forward pass, the network outputs some value which is then compared to the true value (for example 1 in the previous case) and a backward pass is initiated in order to readjust the weights. An artificial neural network might try to model XOR in the following way (after many iterations and backpropagations):

![modelling xor](/imgs/xor_sol.webp)

This is a good approximation of XOR for example because the network could identify when true should be the output and when false should be the output for a given binary input. Bellow I provide my attempt at solving XOR using python’s NumPy library together with the help of several online tutorials.

![jupyter notebooks solution to xor](/imgs/1.png)

- In the first block (using Jupyter notebooks) I initialize all the python libraries that I use. NumPy for math and for handling matrices and vectors.
- My own custom-made Layers classes from pkg.Layers
- The loss (error) function from pkg.loss.
- tqdm for drawing progress bar on terminal when we loop through the network.
- matplotlib for plotting how the error changes with each epoch or iteration through the network.

I then define my activation layer using Tanh (hyperbolic tangent) as my activation function. \*(I do not know why my code does not work well with other activations such as sigmoid or ReLu but I am currently debugging the issue.)

![initializing network, inputs, and true corresponding outputs](/imgs/2.png)

The above code block is used to initialize the inputs, the expected true outputs or “labels”, and the network’s structure. It can be visualized as follows:

![visualizing code block above](/imgs/2.5.png)

I then loop through the network a thousand times (thousand epochs) in order to give it time to adjust its trainable parameters and model the classification problem.

![looping through network](/imgs/3.png)

Finally, we can showcase how the error changes with every iteration:

![loss versus epochs/time/iterations](/imgs/4.png)

In this case the network was able to learn very quickly within the first one-hundred loops. It was able to minimize its error within a short time. However, since the weights are always random at first, you could get other variations of this graph. For example:

![variations of graph](/imgs/5.png)

The error is very low during the first few iterations but then it increases as the network makes more predictions. It’s a case of getting lucky that the initial values were able to model the relationship very accurately. The opposite might also happen. In the references I provide a link to all the tutorials and resources that I used.

## References

---

- [The independent code yt channel](https://www.youtube.com/watch?v=pauPCy_s0Ok)

- [Towards data science article](https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7)

- [Stackoverflow](https://stackoverflow.com/questions/2480650/what-is-the-role-of-the-bias-in-neural-networks)

- [Victor Zhou intro to neural nets](https://victorzhou.com/blog/intro-to-neural-networks/)
