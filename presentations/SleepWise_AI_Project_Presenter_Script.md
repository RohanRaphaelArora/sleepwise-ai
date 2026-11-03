# SleepWise AI — presenter script

**Target delivery:** about 9–10 minutes of speaking, with a short live demo and room for questions. Speak at a steady pace and pause briefly on the results slide.

## Slide 1 — SleepWise AI | ~35 seconds

Hello. My project is SleepWise AI. It uses an artificial neural network to estimate a sleep-quality score from information recorded about sleep and lifestyle. I’ll explain the data, how the system is built, where the neural network fits, what the evaluation showed, and what the limits are. This is an educational prototype. It does not diagnose a sleep disorder or predict someone’s future health.

## Slide 2 — Project question and scope | ~50 seconds

The question is: can a neural network estimate the sleep-quality score recorded in this dataset from the other information recorded with it? This is a regression task because the output is a numeric score. The model can use eleven input characteristics, including age, sleep duration, stress, physical activity, blood pressure, heart rate, steps, gender, occupation, and BMI category. Person ID is just an identifier, so it is excluded. Sleep Disorder is also excluded because it is a recorded outcome that could reveal information about the target. The goal is one estimate for one set of inputs, not a medical conclusion.

## Slide 3 — System architecture | ~70 seconds

This slide shows the full system in two parts. The top row is the training path. First, the project reads the CSV. The data loader checks that the file and expected columns are present. The preparation code cleans the values and splits blood pressure into systolic and diastolic numbers. It removes the ID and leakage fields, then prepares numeric and category inputs. The training module splits the rows by identical input profiles, trains the neural network and comparison models, and evaluates them. The trained model and its preprocessing steps are saved together, alongside metrics and figures.

The bottom row is the live prediction path. A person fills in the Streamlit form. The app validates those values and passes one record to the prediction function. The saved pipeline applies the same scaling and category encoding used during training, then sends the resulting values into the trained ANN. The app displays the estimate. It does not retrain the network each time a field changes. The general sleep information in the app is educational content and is separate from the model output.

## Slide 4 — Dataset and signal | ~55 seconds

The source file has 374 rows and 13 columns. The model uses eleven of those columns as inputs and predicts Quality of Sleep. The scores actually recorded in this file range from 4 to 9, even though the nominal score scale is 1 to 10. There are no missing values. The important limitation is repetition: when the person ID is ignored, 242 rows repeat another row’s input values. That leaves only 109 distinct input combinations. So 374 rows should not be mistaken for 374 independent examples. This affects how we split the data and how cautiously we should interpret the metrics.

## Slide 5 — Preprocessing and evaluation design | ~65 seconds

Before training, the pipeline handles numeric and categorical columns differently. For numeric features, it can fill missing values with the training median and then standardize the values with StandardScaler. For categories, it can fill a missing value with the most common training category and convert each category into one-hot indicators. The transforms are part of the model pipeline, so they are fitted on training rows only.

To reduce leakage from repeated profiles, identical input combinations stay in the same group. We hold out about 20 percent of the groups for the final test and run five-fold grouped cross-validation on the training portion. The feature-importance chart uses permutation importance. It shows how much test error changes when an input is shuffled. That tells us what the fitted model relies on, but it does not prove that an input causes a change in sleep quality.

## Slide 6 — Neural network architecture | ~75 seconds

Here is the ANN itself. After category encoding, the full input has 23 numeric features. The network has two hidden layers, with 64 and 32 units. Each hidden layer uses ReLU, which lets the network learn nonlinear combinations of inputs. The final layer has one linear unit because this is regression and it must output a numeric score.

During training, the network makes a prediction and compares it with the known score using squared error. Backpropagation calculates how each weight contributed to that error and sends that information backward through the layers. Adam uses those gradients to update the weights. L2 regularization discourages overly large weights, and early stopping ends training when validation performance stops improving. The target is standardized during fitting and restored for the application’s displayed score.

The project also includes a small neural network written directly in NumPy. That version makes the forward pass, backpropagation, and Adam updates visible in code, and its gradients are checked against numerical estimates. The deployed network is scikit-learn’s MLPRegressor, while the NumPy version serves as a learning and comparison implementation.

## Slide 7 — Held-out performance | ~55 seconds

The deployed neural network was evaluated on the held-out groups after the model choice was set. The test set contains 86 rows but only 22 distinct input profiles. The mean absolute error is 0.121 points, the root mean squared error is 0.244, and R-squared is 0.942. The training-mean baseline has an RMSE of 1.047, so the ANN performs much better than simply predicting the average score for everyone.

These are results on this one small, repetitive dataset split. They are not a guarantee for new people, and they are not evidence of clinical accuracy. The scatter plot compares the model’s estimated scores with the recorded scores. Points near the diagonal are closer to the actual values.

## Slide 8 — Model comparison and selection | ~55 seconds

The neural network was not the best model in this experiment. Random Forest had the lowest grouped cross-validation RMSE, about 0.237 compared with 0.268 for the ANN. On the held-out test set, Random Forest RMSE was about 0.195 compared with 0.244 for the ANN. We still deploy the neural network because the main course project is specifically about artificial neural networks. That choice was made by project design, not because it had the best result. Reporting the comparison makes the tradeoff visible. With only 22 distinct test profiles, small differences should be treated cautiously.

## Slide 9 — Prediction behavior and data limits | ~65 seconds

In the application, sleep duration is required. If the user adds lifestyle details, the app uses the multivariable ANN. If those details are left out, it uses a separate sleep-duration-only model. Changing inputs changes the estimate because the trained model processes the entered feature values rather than returning a fixed score.

The form accepts age from 1 to 100 and sleep duration from 1 to 10 hours, but those are interface limits. The dataset only covers ages 27 to 59 and sleep durations from 5.8 to 8.5 hours. Predictions beyond the observed ranges are extrapolations. The form also offers Underweight as a BMI choice, but the training file has no Underweight examples. The network therefore cannot learn a specific effect for that category. [If demonstrating the app: change stress or duration, run Analyze Sleep, then toggle optional details to show the two prediction paths.]

## Slide 10 — Takeaways and questions | ~45 seconds

The project shows a complete ANN workflow: load and prepare data, train the network, evaluate it on separated groups, save it, and call it from an application. The network learns useful patterns in this dataset and clearly improves on the mean baseline. Random Forest performs better on the reported comparison, and the small effective sample limits confidence in either model outside this file. The estimate is educational, not a diagnosis or forecast. Thank you. I’m happy to take questions.

## Slide 11 — References | ~15 seconds

These are the selected sources behind the neural-network methods, optimization, early stopping, model comparison, scikit-learn implementation, and dataset. The links on the slide open the source pages. The project report contains the extended bibliography.

## Quick answers if asked

- **Where exactly is the ANN?** train_model.py builds and trains the scikit-learn MLPRegressor. predict.py loads the saved pipeline and calls it for submitted inputs. numpy_mlp.py contains the separate from-scratch teaching implementation.
- **What does backpropagation do here?** It calculates how the output error should change the network’s weights, moving gradients from the output layer back through the hidden layers. Adam applies those weight updates.
- **Why standardize and encode?** Scaling puts numeric features on comparable ranges. One-hot encoding turns categories into numeric indicator columns. The fitted pipeline repeats the same transformations for new inputs.
- **Why group the split?** Many rows repeat the same feature values. Grouping stops an identical profile from appearing in both training and test data, which would make evaluation look too optimistic.
- **Why not deploy Random Forest?** The ANN is the required focus of this neural-network course project. The comparison is still reported honestly.
- **Can this predict a medical condition?** No. It estimates a recorded score from a small dataset and is not a diagnostic or clinical tool.
