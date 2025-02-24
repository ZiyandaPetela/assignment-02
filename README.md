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
* Vlues for x and y
* Model successfully approximated the linear relationship
* Validation results showed strong correlation with expected values

### Project Structure
##### linear_regression_model/
##### ├── src/
  * ├── main.rs      # Main program file
##### ├── Cargo.toml       # Project dependencies
##### ├── README.md        # Project documentation
#### Running the Model
* To train and test the linear regression model, execute:
    * cargo run
#### Expected Output
* The model should learn to predict values close to y = 2x + 1.

## Project Setup
### Prerequisites
#### Ensure you have the following installed:
* Rust (latest stable version)
* Git (for version control and cloning repositories)
* RustRover (for IDE support)

### Clone the Repository
* Run the following command to clone the repository:
   * git clone https://github.com/ZiyandaPetela/linear_regression_model.git
   * cd linear_regression_model
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
### My Personal Challenge and Learning Experience
#### The Core Challenge
* Initially, I faced significant difficulties with this project primarily due to:
   * Limited understanding of the Rust language syntax and concepts
   * Unfamiliarity with the burn library version 0.16.0
   * Challenges in implementing the plotting functionality as required
#### Specific Issues Encountered
##### Code Implementation Struggles:
* Had trouble understanding the tensor operations in burn
* Found it difficult to properly structure the linear regression model
* Struggled with type conversions and compiler errors
* Could not successfully implement the plotting requirements
##### AI Assistance Limitations:
* Discovered that even AI tools had limitations in providing working solutions
* Received code that sometimes didn't compile or work as expected
* Realized that AI couldn't compensate for my fundamental knowledge gaps

#### What I Learned
##### Importance of Fundamentals:
* Having a strong foundation in the programming language is crucial
* Understanding basic concepts cannot be bypassed, even with AI help
* Need to invest time in learning the fundamentals before attempting complex projects
* Gained a deeper understanding of machine learning in Rust.
* Improved debugging skills for Rust compilation errors.
* Learned how to handle dependency conflicts in Rust projects
##### Reality of AI Tools:
* AI is a helpful resource but not a complete solution
* Need to understand the code that AI provides
* Cannot rely solely on AI without personal understanding

  ### I failed to implement the entire solution based on the reasons listed above. The part I didnot finish is plotting the results using the textplots crate.

