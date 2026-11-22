# Reproduction of "How Does Batch Normalization Help Optimization?"

A PyTorch-based reproduction of selected experiments from the NeurIPS 2018 paper **"How Does Batch Normalization Help Optimization?"** by Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry.

## 1. Project Overview

Batch Normalization (BatchNorm) is a widely used technique in deep learning that can improve the training behavior of neural networks.

While BatchNorm was originally introduced to address internal covariate shift, Santurkar et al. (2018) investigate an alternative explanation: BatchNorm improves optimization by making the optimization landscape smoother and gradients more predictable.

The goal of this project is to reproduce selected experiments from the paper and investigate how Batch Normalization affects neural network training.

Specifically, we compare neural networks trained **with and without Batch Normalization**, focusing on:

- Training loss and convergence behavior
- Training and test accuracy
- Sensitivity to different learning rates
- Gradient behavior and optimization stability

The project uses PyTorch and the CIFAR-10 dataset to conduct controlled experiments.

## 2. Original Paper

**Title:** How Does Batch Normalization Help Optimization?

**Authors:** Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry

**Conference:** Advances in Neural Information Processing Systems (NeurIPS), 2018

**Paper:** https://arxiv.org/abs/1805.11604

### Research Question

Why does Batch Normalization improve neural network training?

The paper argues that the benefits of BatchNorm are closely related to optimization: smoother loss landscapes and more predictable gradients can make gradient-based training more effective.

### Reproduction Objectives

This project aims to:

1. Compare the convergence of models with and without BatchNorm.
2. Evaluate the effect of BatchNorm on training and test accuracy.
3. Investigate sensitivity to different learning rates.
4. Examine gradient behavior to better understand optimization stability.
5. Compare experimental observations with the findings reported in the original paper.

The reproduction focuses on selected experiments rather than a complete replication of every result in the paper.

**Project status:** Figure 1 and learning-rate sensitivity experiments have been completed. Gradient analysis is planned.
