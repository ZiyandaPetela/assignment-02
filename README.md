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

## Project Setup
### Prerequisites
#### Ensure you have the following installed:
* Rust (latest stable version)
* Git (for version control and cloning repositories)
* RustRover (for IDE support)
* Visual C++ Build Tools (for Windows users)
#### Installing Visual C++ Build Tools (Windows Only)
* Download and install Visual Studio Build Tools
* Select C++ build tools during installation
* Restart your system if necessary

### Clone the Repository
* Run the following command to clone the repository:
   * git clone https://github.com/ZiyandaPetela/assignment-02.git
   * cd assignment-02
* Verify Rust Installation
* After installing Rust, confirm that it is successfully installed by running the following command:
    * rustc --version
* If Rust is installed correctly, this command should return the installed version.
#### Install Dependencies
* Navigate to the project directory and run:
   * cargo build

## Learning Reflection
### Resources Used
#### Documentation
* Burn Library (v0.16.0) documentation
* Rust official documentation
* RustRover Documentation for IDE usage
#### AI Assistance
* Used AI tools for debugging complex errors
* Leveraged AI for understanding Burn's architecture
* Validated implementation approaches with AI guidance
#### Other Resources
* Stack Overflow for troubleshooting Rust and Burn-related issues
* GitHub discussions for similar implementations
* Technical blogs on Rust machine learning
* Used AI tools to debug errors and understand Burn's implementation.
* Watched YouTube tutorials to educate myself on using Burn effectively.

## Challenges and Solutions
### Technical Challenges
* Dependency Issues: Faced compatibility issues with Burn versions but resolved them by keeping dependencies fixed.
* Compilation Errors: Encountered linking errors on Windows, which were fixed by installing Visual C++ Build Tools.
* Data Formatting: Had to preprocess input data correctly to match the model's expected format
* Difficulty with Burn's tensor operations
* Integration issues with the plotting library

### Learning Outcomes
* Gained a deeper understanding of machine learning in Rust.
* Improved debugging skills for Rust compilation errors.
* Learned how to handle dependency conflicts in Rust projects.
* Recognized the importance of clear documentation for troubleshooting.
* Improved understanding of machine learning fundamentals
* Developed better debugging strategies for Rust projects
##### Even though some errors took time to resolve, this project strengthened my problem-solving abilities and reinforced my knowledge of Rust and AI development.
