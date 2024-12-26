# Salary prediction based on job description in the IT sector

This repository contains the implementation of various machine learning approaches for predicting salaries in IT job postings based on job descriptions and related metadata for ODS Natural Language Processing course (stream 7, autumn 2024)



## Overview

Setting up an optimal salary level in the IT sector presents a significant challenge in a labor market analysis. Here, we address this challenge through the development of an advanced machine learning methodology for salary prediction, achieving a state-of-the-art performance with the R² score of 0.770.

By a systematic investigation of state-of-the-art machine learning architectures, we explored various solutions ranging from gradient boosting frameworks to neural network implementations. The highest prediction quality was observed with a custom setup - integration of CatBoost predictions with a an output from a transformer-based regressor enhanced through the training with Huber loss and having a cross-attention block between represenations of different textual features, resulting in a model capable of sophisticated feature extraction from job descriptions and market indicators. We believe that the source code and the report from this work can be helpful for both academic research in machine learning and practical applications in human resource analytics.


## Project Structure

```
├── baselines/              # Implementation of baseline models
├── data-collection/        # Scripts and tools for data collection
├── experiments/            # Experimental notebooks and scripts
└── predictions_summary/    # Analysis and comparison of model predictions
```

## Results summary


| Experiment | R² score | MAE |
|------------|----------|-----|
| Baselines | | |
| By average | 0.000 ± 0.000 | 0.513 ± 0.002 |
| Bi-GRU-CNN | 0.652 ± 0.012 | 0.288 ± 0.007 |
| CatBoost | 0.734 ± 0.005 | 0.248 ± 0.004 |
| rubert-tiny-turbo (29M) | 0.645 ± 0.027 | 0.289 ± 0.012 |
| Modifications | | |
|------------|----------|-----|
| Double rubert-tiny-turbo | 0.643 ± 0.024 | 0.291 ± 0.013 |
| + Huber loss + TSDAE | 0.657 ± 0.056 | 0.285 ± 0.024 |
|------------|----------|-----|
| rubert-tiny-turbo + Huber loss | 0.655 ± 0.035 | 0.286 ± 0.016 |
| + extra [MASK] pooling | 0.599 ± 0.034 | 0.313 ± 0.015 |
| *+ cross-attention* | *0.671 ± 0.027* | *0.279 ± 0.014* |
|------------|----------|-----|
| multilingual-e5-small (118M) + Huber loss | 0.723 ± 0.024 | 0.254 ± 0.013 |
| *+ cross-attention* | *0.729 ± 0.017* | *0.251 ± 0.009* |
| **+ CatBoost** | **0.770 ± 0.001** | **0.229 ± 0.003** |
| + cross-attention + CatBoost | 0.769 ± 0.014 | 0.229 ± 0.01 |


Performance of the models tested in the study. Metrics are reported as a mean value ± 95% confidence intervals across three random seeds. Overall state-of-the-art results are in **bold**, while the best results for a solo transformer model are in *italics*.




## Participants 
- Eugene Romanov - tg @wallrich
- Dmitrii Shiriaev - tg @dimashiv
- George Besedin - tg @besedin_george
