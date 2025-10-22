# 🛰️ Cross-Sensor Satellite Image Super-Resolution (SEVIRI → MODIS)

This repository contains the DL code and a data visualisation notebook for the MSc thesis project:  
**“Deep Learning Techniques for Satellite Image Downscaling: A Cross-Sensor Super-Resolution Approach for Cloud Mask Products”**  
conducted at the **National and Kapodistrian University of Athens (NKUA)**, STAR MSc Program.

*The data preparation workflow and detailed results will be added soon
---

## 🎯 Project Overview

Satellite instruments face a fundumental trade-off between **spatial** and **temporal** resolution.  
This work explores how a novel dataset can be created to train **Deep Learning (DL)** models
in order bridge this gap by learning cross-sensor transformations between:

- **SEVIRI (MSG)** – high temporal, low spatial resolution (3 km)  
- **MODIS (Terra/Aqua)** – high spatial, low temporal resolution (1 km)

We focus on **cloud mask products**, aiming to generate **high-resolution (HR)** cloud masks from **low-resolution (LR)** SEVIRI inputs using **Super-Resolution (SR)** deep learning models.

---

## 🧠 Objectives

- Develop a **cross-sensor dataset** (MODIS–SEVIRI) with temporal and spatial alignment  
- Train and compare **Super-Resolution DL models** (SRCNN, SRGAN)  
- Evaluate models against classical interpolation baselines (bicubic)  
- Assess improvements using **PSNR**, **SSIM**, and qualitative visualization  

---

## 🗂️ Dataset

A novel dataset of initially over **60,000 temporally and spatially matched** SEVIRI–MODIS pairs was created:
- **Input:** choose from SEVIRI 11 spectral channels + cloud mask (32×32 px)
- **Target:** MODIS cloud mask (128×128 px)
- Temporal matching within a **15-minute window**

---

## ⚙️ Models Implemented

- **SRCNN** – Super-Resolution Convolutional Neural Network  
- **SRGAN** – Super-Rsolution Generative Adversarial Network  

Each model was trained to achieve **4× spatial resolution enhancement**.

---

## 📈 Results

Both DL models perform comparably to bicubic interpolation in terms of **PSNR** and **SSIM**, and a SRGAN experiment showed consistently significant improvements in SSIM compared to bicubic interpolation, achieving visually sharper and more accurate cloud mask predictions.  
These results demonstrate the feasibility of **cross-sensor SR** for near-real-time atmospheric monitoring.

---

## 🧩 Applications

- Weather forecasting  
- Renewable energy prediction  
- Climate and atmospheric research  
- Disaster risk reduction  

---

## 🧑‍💻 Author

**Angelos Georgakis**  
M.Sc. in Space Technologies, Applications and SeRvices (STAR)  
[National and Kapodistrian University of Athens](https://en.uoa.gr/)

---

## 📚 Citation

If you use this work, please cite as:

> Georgakis, A. (2025). *Deep Learning Techniques for Satellite Image Downscaling: A Cross-Sensor Super-Resolution Approach for Cloud Mask Products*. MSc Thesis, NKUA.

---

## 📜 License

This project is released under the **MIT License**. 
