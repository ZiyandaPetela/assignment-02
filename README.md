# Linear Regression Model Using Rust and Burn
## Introduction
* This project uses the Rust programming language and the Burn library (v0.16.0) to build a linear regression model that predicts values for the function y = 2x + 1.
* The implementation simulates real-world settings using synthetic data with extra noise and teaches basic Rust machine learning ideas.

## Approach

### Data Generation and Preprocessing
* Created synthetic (x, y) pairs following y = 2x + 1
* Added Gaussian noise to simulate real-world data variation
* Implemented data normalization to improve training stability
  
### Model Architecture
* Developed a simple linear regression model using Burn's neural network modules
* Implemented Mean Squared Error (MSE) as the loss function
* Used Burn's optimization capabilities for model training
  
### Training Process
* Split data into training and validation sets
* Implemented iterative training with gradient descent
* Monitored loss values to ensure proper convergence
* Adjusted learning rate and batch size for optimal performance
  
## Results and Evaluation

### Model Performance
* Final training loss: [Your final loss value]
* Model successfully approximated the linear relationship
* Validation results showed strong correlation with expected values
### Visualization
#### Used textplots crate to create visual representations of:
* Training data distribution
* Model predictions vs. actual values
* Loss convergence over training iterations

