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
}