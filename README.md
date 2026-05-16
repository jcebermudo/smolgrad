# Kapi

Kapi is a tiny deep learning framework in TypeScript. It's meant to be a learning tool for understanding how to build and train simple neural networks.

This started off as a simple reimplementation of Andrej Karpathy's micrograd, but has evolved into a framework for deep learning tasks in TypeScript with PyTorch-like syntax.

## Using partial derivatives

The heart of backpropagation is partial derivatives. You need backprop to know how much you need to adjust your model's parameters for better results. 

Remember, derivatives are basically slopes, you want to know how much something changes at a specific point.

Let's say you have a chain of functions. Let's start with the variables `a` and `b` .

`a = 2`
`b = 3`

Let's define these functions:
`c = a + b`
`d = c * 2`
`e = d ^ 2`
`f = relu(e)`

At the highest level, neural networks can be compared to deeply nested functions. When you compute for its loss, you want to know how much each the functions contribute to the final output, using the slope to know how to adjust your network's parameters.

Now, as an example, in the given functions and variables provided, let's perform chain rule on partial derivatives to see how much `a` contributes to the final output of `f` through a slope.

The plan is to go backwards, get the partial derivative of each function, and multiply all of them at the end:
`d(f)/d(a) = d(f)/d(e) * d(e)/d(d) * d(d)/d(c) * d(c)/d(a)`

`f -> d(f)/d(e) = 1`
`e -> d(e)/d(d) = 2 * d = 20`
`d -> d(d)/d(c) = 2`
`c -> d(c)/d(a) = 1`

let's get the slope!
`1 * 20 * 2 * 1 = 40`

