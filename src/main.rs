use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct LinearRegression<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> LinearRegression<B> {
    pub fn new(device: &B::Device) -> Self {
        // Initialize weights and bias using proper tensor creation
        let weight = Tensor::zeros([1], device);
        let bias = Tensor::zeros([1], device);

        Self { weight, bias }
    }

    // Forward pass implementation
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Clone and reshape weight for matrix multiplication
        let weight = self.weight.clone().reshape([1, 1]);
        // Perform matrix multiplication
        let wx = x.matmul(weight);
        // Add bias (broadcast automatically)
        wx + self.bias.clone().reshape([1, 1])
    }

    // Mean Squared Error loss function
    pub fn mse_loss(&self, predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 0> {
        let diff = predictions - targets;
        // Use mul for element-wise multiplication
        let squared_diff = diff.clone().mul(diff);
        // Reduce to scalar by taking mean and ensuring 0 dimensions
        squared_diff.mean().reshape::<0, [usize; 0]>([])
    }
}

// Model record for training
#[derive(Module, Debug)]
pub struct ModelRecord<B: Backend> {
    pub model: LinearRegression<B>,
}

fn main() {
    println!("Linear Regression Model defined successfully!");
}