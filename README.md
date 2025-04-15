# Banknote Classification System Using NuMaker-M032KG

This project implements a banknote classification system based on RGB light intensity sensing and a neural network model. It utilizes three photoresistors with color filters and a Multilayer Perceptron (MLP) deployed on the NuMaker-M032KG microcontroller platform.

<img src="https://github.com/user-attachments/assets/4ea6764f-7dda-4cb0-b275-6f67adf0eb6f" width="750"/>

---

## Overview
- [Hardware Platform](#hardware-platform)
- [Sensor and Circuit Design](#sensor-and-circuit-design)
- [Model Architecture](#model-architecture)
- [System Workflow](#system-workflow)
- [Processing Pipeline](#processing-pipeline)
- [System Diagram](#system-diagram)
- [Demonstration Video](#demonstration-video)

---

## Hardware Platform

### NuvoTon NuMaker-M032KG Development Board  
[Product Page](https://direct.nuvoton.com/tw/numaker-m032kg)

<img src="https://github.com/user-attachments/assets/53c2646d-d427-4818-993f-16b76a3c903f" width="500"/>

---

## Sensor and Circuit Design

Key components include:
- RGB photoresistors with color filters (R, G, B)
- White LED (active illumination source)
- Voltage divider network
- ADC interface to M032KG microcontroller

Working Principle:
- LED provides active lighting toward the target banknote.
- Reflected light passes through RGB filters to respective photoresistors.
- Analog signals are converted to digital values via onboard ADC.

<img src="https://github.com/user-attachments/assets/07924d79-20af-4fce-878a-5adcfb9e83ce" width="500"/>

---

## Model Architecture

- Model Type: Multilayer Perceptron (MLP)
- Input: Resampled RGB time-series signals
- Output: Banknote denomination class

---

## System Workflow

### RGB Light Sensing
- Colors are detected using a combination of filtered light responses.
- Each photoresistor detects specific spectrum reflections (R, G, B).
- ADC readings reflect intensity levels for each channel.

---

## Processing Pipeline

1⃣ **Initialization Phase**  
Baseline RGB light levels are measured under ambient conditions.

2⃣ **Detection Start**  
Upon detecting significant light change, recording begins.

3⃣ **Detection End**  
A second shift in values signals the object has exited. Stop recording.

4⃣ **Signal Resampling**  
Normalize the captured data to a fixed time length for consistent input.

5⃣ **Model Inference**  
Feed resampled data into the MLP for prediction.

6⃣ **Result Output**  
Display the predicted denomination locally.

---

## System Diagram

<img src="https://github.com/user-attachments/assets/c9f31a80-07d1-4316-83b5-332495b21d1b" width="600"/>

---

## Demonstration Video

[![Banknote Recognition Demo](https://img.youtube.com/vi/qb9uLU0ng0Y/0.jpg)](https://www.youtube.com/watch?v=Xpk8Segaels)
