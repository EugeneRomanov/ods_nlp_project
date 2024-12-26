# Salary prediction based on job description in the IT sector

This repository contains the implementation of various machine learning approaches for predicting salaries in IT job postings based on job descriptions and related metadata for ODS Natural Language Processing course (stream 7, autumn 2024)



## Overview

Determining optimal compensation in the Information Technology sector presents a significant challenge in contemporary labor market analysis. This research addresses this challenge through the development of an advanced machine learning methodology for salary prediction, achieving a coefficient of determination (R² score) of 0.772, which represents a substantial advancement in predictive accuracy.

The research methodology encompasses a systematic investigation of state-of-the-art machine learning architectures, ranging from gradient boosting frameworks to neural network implementations. The principal contribution lies in the novel integration of CatBoost predictions with a custom transformer architecture, enhanced through the implementation of Huber loss and cross-attention mechanisms, resulting in a model capable of sophisticated feature extraction from job descriptions and market indicators.

This repository presents a comprehensive documentation of our research methodology and implementation, demonstrating the practical application of contemporary machine learning techniques to labor market analysis. The work offers significant implications for both academic research in applied machine learning and practical applications in human resource analytics.


## Project Structure

```
├── baselines/          # Implementation of baseline models
├── data-collection/    # Scripts and tools for data collection
├── experiments/        # Experimental notebooks and scripts
└── predictions_summary/# Analysis and comparison of model predictions
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
| Double rubert-tiny-turbo | 0.643 ± 0.024 | 0.291 ± 0.013 |
| + Huber loss + TSDAE | 0.657 ± 0.056 | 0.285 ± 0.024 |
|------------|------------|------------|
| rubert-tiny-turbo + Huber loss | 0.655 ± 0.035 | 0.286 ± 0.016 |
| + extra [MASK] pooling | 0.599 ± 0.034 | 0.313 ± 0.015 |
| *+ cross-attention* | *0.671 ± 0.027* | *0.279 ± 0.014* |
| --- | | |
| multilingual-e5-small (118M) + Huber loss | 0.723 ± 0.024 | 0.254 ± 0.013 |
| *+ cross-attention* | *0.729 ± 0.017* | *0.251 ± 0.009* |
| **+ CatBoost** | **0.770 ± 0.001** | **0.229 ± 0.003** |
| + cross-attention + CatBoost | 0.769 ± 0.014 | 0.229 ± 0.01 |


Performance of the models tested in the study. Metrics are reported as a mean value ± 95% confidence intervals across three random seeds. Overall state-of-the-art results are in **bold**, while the best results for a solo transformer model are in *italics*.


![image](https://github.com/user-attachments/assets/412647b1-9e25-42da-8400-82d4bd64e91b)



## Participants 
- Eugene Romanov - tg @wallrich
- Dmitrii Shiriaev - tg @dimashiv
- George Besedin - tg @besedin_george
