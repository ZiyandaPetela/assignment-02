use rand::{rng, Rng}; // Updated function

fn generate_data(samples: usize) -> Vec<(f32, f32)> {
    let mut rng = rng(); // Use new function
    let mut dataset = Vec::new();

    for _ in 0..samples {
        let x = rng.random_range(-10.0..10.0);
        let noise: f32 = rng.random_range(-0.5..0.5); // Adding noise
        let y = 2.0 * x + 1.0 + noise;
        dataset.push((x, y));
    }

    dataset
}

fn main() {
    let dataset = generate_data(100);

    for (x, y) in dataset.iter().take(10) { // Print first 10 values
        println!("x: {:.2}, y: {:.2}", x, y);
    }
}

