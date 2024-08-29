# Topic: Dimensionality reduction of latent space for interpretability assessment of neural network models

## My notes

## Preparation

- get familiar with Seaborn, plotly for scatter plots

- get familiar with microconda, micromamba for environment management

- get familiar with numpy for data loading, saving

- get familiar with pandas for data loading, saving, basic manipulation, plotting via seaborn

- get familiar with pytorch for loading a trained network

- create python 3.10 environment with UMAP, T-SNE/opt-SNE, Trimap, PacMAP, scikit-learn, seaborn/plotly, pandas

- download MNIST

## Work plan

1. Reproduce (figure 2 wang et al 2021) results with UMAP, T-SNE, Trimap on MNIST

                - UMAP - tune number of neighbours manually

                - T-SNE

                                - tune perplexity manually

                                - use opt-SNE https://github.com/omiq-ai/Multicore-opt-SNE

                - Trimap - tune number of inliers manually

                - PacMAP with defaults + sensitivity analysis

2. Apply to Feedforward network latent space (from Hannes Gubler)

                - test different layers (second, third, fourth)

                - plot scatter with coloring using class labels (mRS 3 months positive or negative) or continuous features (NIHSS at admission, acute glucose level)

3. Apply to Graph neural network latent space (from Jaume Banus Cobo)

                - test different depth (bottleneck, others)

                - plot scatter with class labels (ACDC disease category), continuous features (ejection fraction, average strain...)

4. (bonus) Train random forest classifier on dimensionaly-reduced space (2D, 3D, 4D) and see how good that is, and how classification performance varies depending on dimRed parameters.

                - whole sample dimRed (cheating )

                - cross-val dimRed (out-of-sample application of dimRed)
