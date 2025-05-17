## 🧭 IGG

IGG: Image Generation Informed by Geodesic Dynamics in Deformation Spaces (https://arxiv.org/pdf/2504.07999).

## 💡 Disclaimer

This code is only for research purpose and non-commercial use only, and we request you to cite our research paper if you use it:  
**IGG: Image Generation Informed by Geodesic Dynamics in Deformation Spaces**  
Nian Wu, Nivetha Jayakumar, Jiarui Xing, and Miaomiao Zhang. *Information Processing in Medical Imaging (IPMI 2025)*.

@article{wu2025igg,  
  title={IGG: Image Generation Informed by Geodesic Dynamics in Deformation Spaces},  
  author={Wu, Nian and Jayakumar, Nivetha and Xing, Jiarui and Zhang, Miaomiao},  
  journal={arXiv preprint arXiv:2504.07999},  
  year={2025}  
}

## 📌 Setup

The main dependencies are listed below, the other packages can be easily installed with "pip install" according to the hints when running the code.

* python=3.10
* pytorch=2.0.0
* cuda11.8
* matplotlib
* numpy
* SimpleITK
* LagoMorph

Tips:
LagoMorph contains the core implementation for solving the geodesic shooting equation (i.e., the EPDiff equation) under the LDDMM framework.
The code repository is available at: https://github.com/jacobhinkle/lagomorph


## 🚀 Usage

Below is a **QuickStart guide** on how to use **IGG** for network training and testing.

---

### 🔹 **Phase 1: Autoencoder-Based Registration Network**

This module learns **latent representations of geodesic paths** in deformation space using an autoencoder registration network.

**To train and test this module**, run:

```bash
bash IGG/bash/IGG_AutoEncoder_Train.sh
bash IGG/bash/IGG_AutoEncoder_Test.sh

Required Input: Paired source and target images. 
```


### 🔹 **Phase 2: Latent Geodesic Diffusion Model**

This module learns the **distribution over latent geodesic trajectories obtained from Phase 1**, using a conditional diffusion process..

**To train and test the diffusion model**, run:

```bash
bash IGG/bash/IGG_DiFuS_Train.sh
bash IGG/bash/IGG_DiFuS_Test.sh

Required Input: A template image and its corresponding text instruction.
```
