# Yahtzee Reinforcement Learning
This repo is for the final project for the *Case Studies in Machine Learning* course in UT Austin's Master of Science in Artificial Intelligence.

## Organization
This project is configured using the `pape-lab` infrastructure, which provides a hookup to Weights & Biases (much better than using tensorboard!) and lets me farm jobs out to either my local GPU or to Amazon SageMaker.

As such, everything (including the inner dev loop) is organized around creating a docker container. The primary dockerfile can be found in [Dockerfile.wandb](./Dockerfile.wandb). This is required for running in SageMaker, but has the added benefit of reproducibility or sharing in the future.

So far in this project I've been building the infrastructure for doing RL on Yahtzee from the ground up. 

## A) Dice Maximizer 
A basic setup for doing REINFORCE on the problem of "maximizing the face value of all dice". This was primarily to get my dev environment setup, ensure the infrastructure for OpenAI Gymnasium, PyTorch Lightning, W&B, and AWS was working properly.

This can be run with `cd src && ./run_dice_maximizer.sh`.

You'll see the model very quickly learns it should keep dice that are above 4, which is pretty much the optimal strategy. It scores around the theoretical limit for this problem, which is ~23.5.

## B) Supervised Scorer
Here I tried to do supervised learning on the subset problem of trying to figure out which categories would be the best given a set of dice. Part of the thought process here was that later in the project I may want to do some form of behavioral cloning before moving on to RL for fine-tuning.

This can be run with `cd src && ./run_supervised_scorer.sh`.

It's not particularly interesting, it also learns this trivial problem fairly quickly, and I so far have not used it in a later phase.

## C) Single Turn Score Maximizer
This is where things start to get interesting. The entire focus of this project is doing pure RL for the single-turn problem.

Basically, the question is: "only training on a single turn of roll, roll, score; how good can we do overall at the game of Yahtzee?"

These can be run with `cd src && ./run_single_turn_rl.sh`.

So far best runs are scoring in the 160 range.