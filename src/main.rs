use rand::{rng, Rng};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

// Data generation function from code #1
fn generate_data(samples: usize) -> Vec<(f32, f32)> {
    let mut rng=rng();
    let mut dataset = Vec::new();
    for _ in 0..samples {
        let x = rng.random_range(-10.0..10.0);
        let noise = rng.random_range(-0.5..0.5);
        let y = 2.0 * x + 1.0 + noise;
        dataset.push((x, y));
    }
    dataset
}

// Model definition from code #2
#[derive(Module, Debug)]
pub struct LinearRegression<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> LinearRegression<B> {
    pub fn new(device: &B::Device) -> Self {
        let weight = Tensor::zeros([1], device);
        let bias = Tensor::zeros([1], device);
        Self { weight, bias }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let weight = self.weight.clone().reshape([1, 1]);
        let wx = x.matmul(weight);
        wx + self.bias.clone().reshape([1, 1])
    }

    pub fn mse_loss(&self, predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 0> {
        let diff = predictions - targets;
        let squared_diff = diff.clone().mul(diff);
        squared_diff.mean().reshape::<0, [usize; 0]>([])
    }
}
pub fn train(&mut self, x: &Vec<f32>, y: &Vec<f32>, epochs: usize, learning_rate: f32) {
        let n = x.len() as f32;

        for epoch in 0..epochs {
            // Dummy forward pass (replace with proper tensor logic)
            let predictions: Vec<f32> = x.iter().map(|&x| 2.0 * x + 1.0).collect();

            // Compute loss (MSE)
            let loss: f32 = predictions.iter()
                .zip(y.iter())
                .map(|(&pred, &target)| (pred - target).powi(2))
                .sum::<f32>() / n;

            // Monitor training progress
            if epoch % 100 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, loss);
            }
        }
}
// Model record from code #2
#[derive(Module, Debug)]
pub struct ModelRecord<B: Backend> {
    pub model: LinearRegression<B>,
}

fn main() {
    let dataset = generate_data(100);
    for (x, y) in dataset.iter().take(10) {
        println!("x: {:.2}, y: {:.2}", x, y);
    }
   let x_train: Vec<f32> = dataset.iter().map(|(x, _)| *x).collect();
    let y_train: Vec<f32> = dataset.iter().map(|(_, y)| *y).collect();

    println!("\nTraining model...");
    let mut model = LinearRegression::new();  
    model.train(&x_train, &y_train, 1000, 0.01);
} 
}
