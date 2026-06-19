# **MACE-FT-Tutorial**  
*A Practical Guide to Fine-Tuning Universal Machine-Learned Interatomic Potentials (U-MLIPs)*  

## **Overview**  
Universal machine-learned interatomic potentials (U-MLIPs) have demonstrated broad applicability across various atomistic systems. However, fine-tuning is often necessary to achieve optimal accuracy for specific applications. Despite the increasing availability of U-MLIPs and their fine-tuning methodologies, there remains a lack of systematic guidance on refining these models effectively for materials science problems.

This tutorial provides a **structured, step-by-step guide** to fine-tuning U-MLIPs for computational materials modeling. Using **MACE-MP-03b, MACE-MPA-0, and MACE-OMAT-0** as representative models, we cover essential aspects of the fine-tuning workflow, including:

- **Dataset preparation and preprocessing**  
- **Hyperparameter selection**  
- **Model training and validation**  
- **Case studies on metals, organics, and surfaces**  

These case studies illustrate the impact of fine-tuning on predictive accuracy and model transferability.

⚠ **Note:** *MACE-OMAT-0 is licensed under ASL and is strictly for academic use.*

## **Why Use This Tutorial?**  
✔ **Hands-on learning** – Provides practical code examples for easy implementation  
✔ **Accessible for beginners** – Ideal for researchers new to MLIP fine-tuning  
✔ **Performance optimization** – Uses novel techniques to accelerate model inference  
✔ **Application-focused** – Showcases diverse and compelling examples  

## **Getting Started**  
To get started, clone the repository and follow the provided Jupyter Notebooks:

```bash
git clone https://github.com/YangshuaiWang/mace-ft-tutorial.git
cd mace-ft-tutorial
```

Refer to [`setup.md`](./setup.md) for installation instructions and dependencies.

## **Runnable Paper Examples**

Two paper case studies have self-contained command-line examples in this repo:

- **A. Li10GeP2S12**: [`examples/LiGePS-SSE-PBE`](./examples/LiGePS-SSE-PBE)
  prepares LGPS `extxyz` train/test data and fine-tunes MACE with the paper's
  optimized LGPS setting (`forces_weight=100`, `lr=0.001`,
  `ema_decay=0.999`).
- **D. Graphene-water interface**:
  [`examples/graphene-water`](./examples/graphene-water) fine-tunes on the
  prepared interface dataset, runs 300 K ASE/Langevin MD, and analyzes the
  oxygen density profile used to estimate the graphene-water distance.

After activating an environment with the required Python dependencies, quick
smoke tests are:

```bash
cd examples/LiGePS-SSE-PBE
python3 prepare_ligps_data.py --max-frames 20

cd ../graphene-water
python3 run.py --config graphene-128.xyz --steps 10 --optimize-steps 0 --device cpu
```


## **Citations**  
If you use the foundation models directly, please cite:  

```bibtex
@article{batatia2023foundation,
      title={A foundation model for atomistic materials chemistry},
      author={Ilyes Batatia and Philipp Benner and Yuan Chiang and Alin M. Elena and Dávid P. Kovács and Janosh Riebesell and Xavier R. Advincula and Mark Asta and William J. Baldwin and Noam Bernstein and Arghya Bhowmik and Samuel M. Blau and Vlad Cărare and James P. Darby and Sandip De and Flaviano Della Pia and Volker L. Deringer and Rokas Elijošius and Zakariya El-Machachi and Edvin Fako and Andrea C. Ferrari and Annalena Genreith-Schriever and Janine George and Rhys E. A. Goodall and Clare P. Grey and Shuang Han and Will Handley and Hendrik H. Heenen and Kersti Hermansson and Christian Holm and Jad Jaafar and Stephan Hofmann and Konstantin S. Jakob and Hyunwook Jung and Venkat Kapil and Aaron D. Kaplan and Nima Karimitari and Namu Kroupa and Jolla Kullgren and Matthew C. Kuner and Domantas Kuryla and Guoda Liepuoniute and Johannes T. Margraf and Ioan-Bogdan Magdău and Angelos Michaelides and J. Harry Moore and Aakash A. Naik and Samuel P. Niblett and Sam Walton Norwood and Niamh O'Neill and Christoph Ortner and Kristin A. Persson and Karsten Reuter and Andrew S. Rosen and Lars L. Schaaf and Christoph Schran and Eric Sivonxay and Tamás K. Stenczel and Viktor Svahn and Christopher Sutton and Cas van der Oord and Eszter Varga-Umbrich and Tejs Vegge and Martin Vondrák and Yangshuai Wang and William C. Witt and Fabian Zills and Gábor Csányi},
      year={2023},
      eprint={2401.00096},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
```

If you find this repository useful, please consider citing:
```
@misc{wang2025maceft,
  title={MACE-FT-Tutorial},
  author={},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/YangshuaiWang/mace-ft-tutorial}},
  year={2023}
}
```


<!-- ## **Contributors**  
**Author:** Yangshuai Wang, XXX, XXX, XXX, XXX or Team RBMD  
For questions and discussions, feel free to open an issue or reach out. -->
