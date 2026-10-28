# Viva preparation (Artificial Neural Networks and Deep Learning)

Numbers below come from `reports/model_metrics.json`. If you retrain, re-check them.

**1. What is the problem and why a neural network?**
Estimate the recorded sleep-quality score (4 to 9 in the data) from 11 characteristics. It is non-linear regression; a feed-forward network can model non-linear interactions. I also compared it with simpler models to see whether the extra complexity pays off.

**2. Describe your architecture.**
23 → 64 → 32 → 1: 23 inputs after encoding (8 numeric + one-hot of gender, occupation, BMI), hidden layers of 64 and 32 ReLU units, one linear output. 3,649 trainable parameters.

**3. Why ReLU? Why a linear output?**
ReLU does not saturate for positive inputs, so gradients flow better than with sigmoid/tanh, and it is cheap. The target is an unbounded real number, so the output layer has no squashing function.

**4. Explain back-propagation.**
Forward pass computes activations and the loss. Backward pass applies the chain rule layer by layer: the output error δ = (ŷ − y)/n, then the weight gradient is (previous activation)ᵀ·δ, and δ is pushed back through Wᵀ and multiplied by the ReLU derivative (1 where z > 0, else 0). I implemented this in `numpy_mlp.py` and verified it against numerical gradients (relative error about 1e-7).

**5. What optimiser did you use and why?**
Adam: it keeps running averages of the gradient and squared gradient to give each weight an adaptive step size, and works well with little tuning. Learning rate 0.001.

**6. How did you fight overfitting?**
L2 weight penalty (α = 0.001), early stopping on an internal validation slice (stopped at epoch 86), a modest network, cross-validation, and a held-out test set. There is still a gap: training R² 0.987 vs test R² 0.942.

**7. Why scale the inputs and the target?**
Features on different scales make the loss surface badly conditioned and slow or destabilise gradient descent. Numbers are standardised, categories one-hot encoded, and the target standardised for training then converted back.

**8. How did you initialise weights?**
scikit-learn uses a Glorot-style scheme by default; in my NumPy network I used He initialisation (variance 2 / fan_in), which suits ReLU.

**9. What is early stopping?**
Training halts when the validation score stops improving for 30 epochs, limiting how closely the network fits noise.

**10. What does the loss curve show?**
Training loss falls steeply, then flattens; the validation R² (dashed) plateaus. It shows the network converging and early stopping triggering. The validation slice is not grouped, so it is optimistic.

**11. Your neural network is not the best model. Why deploy it?**
This is a neural-network course and I fixed the deployed model before looking at test results. I report the Random Forest as best by cross-validation (RMSE 0.237 vs 0.268) and on test (RMSE 0.195 vs 0.244). On a small tabular dataset trees often do as well or better. `config.DEPLOYED_MODEL = "auto"` would deploy the CV winner instead.

**12. How do you know the evaluation is trustworthy?**
Preprocessing is inside the pipeline (fit on training only); the target, ID and Sleep Disorder are excluded; the test set was used once; and rows with identical inputs are kept on the same side of the split. Even so, only 22 distinct test combinations exist, so the estimate is uncertain.

**13. What is the issue with repeated rows?**
242 of 374 rows repeat another row's inputs. In a random split 93% of test rows would have a twin in training, giving R² 0.992 instead of 0.942.

**14. Why RMSE, MAE and R²?**
MAE is the average miss in score points; RMSE penalises large misses more; R² compares with always predicting the mean (the baseline has R² -0.071 on test).

**15. What is the bias-variance trade-off here?**
Linear regression has high bias (R² 0.850 on test). The network and forest have lower bias but higher variance; regularisation and early stopping reduce variance.

**16. What does permutation importance mean?**
Shuffle one input column on the test rows and see how much RMSE rises. It shows what the model relies on, not what causes sleep quality. The network's top inputs: Sleep Duration (0.31), Occupation (0.19), Diastolic BP (0.19), Stress Level (0.18), Gender (0.15).

**17. Why does the network rely on gender and occupation more than the forest?**
Not tested. A plausible reason is memorising patterns tied to specific combinations in a very small dataset. I would not claim it is a real effect.

**18. Is the data real?**
Its provenance is not documented in the file, and its structure (109 distinct input combinations in 374 rows) suggests templated or synthetic data. I describe it as a public dataset of unverified provenance, and make no clinical claims.

**19. Could this be used medically?**
No. It estimates a recorded score; it does not diagnose or forecast, and associations are not causes.

**20. What did you do yourself?**
*(Your own answer. See the contribution section of the report.)*

**23. Why compare sleep duration only with all features?**
To avoid assuming that more features are better, and to check whether duration alone supports predictions. Duration alone gives test R² 0.72, all clues 0.94. Both are limited by a small, highly structured dataset.

**24. Why does the duration-only model give stepped predictions?**
Its best candidate by cross-validation is a Decision Tree, which predicts one constant value per leaf, so the output changes in steps.

**25. What does the Potential Effects page do, and is it ML?**
It is not ML. It paraphrases CDC, NHLBI and AASM/SRS guidance with source links, labelled as educational, with no probabilities. The model never predicts effects because the dataset has no effect labels.

**21. What would you improve?**
External validation data, prediction intervals (conformal prediction or deep ensembles), nested-CV hyper-parameter search, embeddings for occupation, fairness checks.

**22. How is the app deployed and what happens if the model file is incompatible?**
GitHub + Streamlit Community Cloud. Artefacts are committed; if the model is missing or built with a different scikit-learn version the app retrains once from the bundled CSV.
