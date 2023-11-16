# TensorFlow Lite Benchmark

<!-- ![TensorFlow Lite Logo](https://www.gstatic.com/webp/gallery/2.jpg) -->

This repository provides tools and resources to benchmark TensorFlow Lite models on various hardware platforms specialy arm based embedded systems, making it easier for developers and researchers to measure the performance of their models.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Benchmarking](#benchmarking)
  - [Custom Models](#custom-models)
- [Contributing](#contributing)
- [License](#license)

## Introduction

TensorFlow Lite is an optimized machine learning framework for mobile and embedded devices. Benchmarking TensorFlow Lite models is crucial for understanding their runtime performance on various hardware, including CPUs, GPUs, and accelerators like Edge TPUs. This repository is designed to help you evaluate and compare the performance of your TensorFlow Lite models across different devices and configurations.

## Getting Started

Follow these steps to set up the benchmarking environment.

### Prerequisites

- Python 3.6 or higher
- TensorFlow Lite (TFLite) library
- TensorFlow (optional, for TensorFlow models)
- Supported hardware platforms (e.g., Android devices, Edge TPUs, Raspberry Pi)

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Mohammadakhavan75/tflite-benchmark.git
   ```
1.1 Installing required packages

1.1.1 If you're running Debian Linux or a derivative of Debian please use this:
  
  ```bash
  echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update
  sudo apt-get install python3-tflite-runtime
  ```

1.1.1 For other systems you can use:
  ```bash
  pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
  pip install -r requirements.txt
  ```
  