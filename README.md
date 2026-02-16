<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [flowerNeuralNetwork](#flowerneuralnetwork)
  - [Overview](#overview)
  - [About me](#about-me)
  - [State of the art](#state-of-the-art)
  - [A simple example problem](#a-simple-example-problem)
  - [Visualization](#visualization)
  - [Neural Network Solution](#neural-network-solution)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# flowerNeuralNetwork

This is a learning exercise to teach myself everything I could possibly want to know about neural networks, from the math behind them to how to create one myself and use it solve my own problem.

I start with a simple dataset and problem to solve.

I use a neural network to automate the task a farmer solves manually.

A data table of flower measurements is created and then visualized using a graph. I learnt how this manual process sets the stage for automation.

---

## Overview

Neural Networks is a really cool field from mathematics and computer science. They sit at the core of artificial intelligence that has taken the world by storm.
Lately, on your phone you can kind ask it a question or upload a photo and it knows who's in it or even what's in it. You may be curious you know how is this is possible. 

These capabilites are being driven by neural networks. This code exercise showcases what
I know about them.

## About me

I have a BSc in Maths which is the sort of background knowledge that helped me to understand the underlying maths. I cover the linear algebra and calculus I need to understand and include in own python code.

##  State of the art

Neural Networks allow computers to actually outperform a human at recognizing stuff in pictures. A competition between a human and an AI transformer trying to identify specific breeds of dogs would illistrate how effective AI can be.
My code will show how you actually train your model using back propagation.

## A simple example problem

A farmer likes to measure everything around her. She was growing some flowers one day and realized she hadn't measured them so she decided that this day was the perfect day to take out her ruler and take some measurements. She has two types of flowers; red ones and blue
ones.
She takes out her ruler, lays the flowers down. She starts with the red flower and plucks a petal off, lays it down, then measures it's width and length. What she needs now is a table to record all of her data.

| Petal Length | Petal Width | Flower Color |
|---|---|---|
| 3 | 1.5 | Red |
| 2 | 1 | Blue |
| 4 | 1.5 | Red |
| 3 | 1 | Blue |
| 4.5 | 1 | ? |

She measured the dimensions of the last petal from a flower but it looks like she forgot to measure the color or just note down the color of this last flower. She's a little upset there's a bit of a mystery here and she cannot be happy unless her data set is complete so she has to think about this problem. There's a few ways that she could solve this that immediately come to her mind she could compare these numbers to the other numbers and maybe um if they're similar she kind of assumes that red flower measurements are all fairly similar. She has a better idea. She's going to graph them.

## Visualization

The farmer then graphs these measurements on a scatter plot:

x is Blue
● is Red

```
    Width (cm)
         2.0|
         1.5|        ●    ●
         1.0|   x    x      ?
         0.0|____________________
           0    2    3    4    5  Length (cm)
```

By plotting the petal length and width, she can visually identify that the mystery flower's measurements (4.5, 1) cluster closer to the red flowers, allowing her to confidently classify it as red.

She does have a bit of doubt but she has a friend and her friend is a computer and the computer has a brain which could do the same task for her if she taught the computer how to do it and we call that brain a neural network.

## Neural Network Solution

A neural network with input and output layers can solve this classification problem:

```
    
    ●
   / \            
  /   \           
 ●     ●
    
L      W
```

The neural network learns to map petal measurements to flower color through training, automating the farmer's classification task at scale.

This could let the computer automate that task that our farmer had to do and the computer can do it a lot faster than her. Let's say there were 10,000 flowers the computer could crunch through that data very quickly and give her an estimate much faster than she could do herself.