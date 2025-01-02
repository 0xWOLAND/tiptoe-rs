use anyhow::Result;
use ndarray::{Array1, Array2, Axis, ArrayView1};
use ndarray_stats::DeviationExt;
use rand::prelude::*;
use simplepir::Matrix;
use std::collections::HashMap;

struct RollingMean {
    current_mean: Array1<f64>,
    n_samples: u64,
}

impl RollingMean {
    fn new(first_sample: Array1<f64>) -> Self {
        RollingMean {
            current_mean: first_sample,
            n_samples: 1,
        }
    }

    fn accumulate(&mut self, new_sample: &Array1<f64>) {
        let mut increment: Array1<f64> = &self.current_mean - new_sample;
        increment.mapv_inplace(|x| x / (self.n_samples + 1) as f64);
        self.current_mean -= &increment;
        self.n_samples += 1;
    }
}

pub fn get_centroids(matrix: &Matrix) -> Result<Vec<Vec<u64>>> {
    let n_samples = matrix.nrows;
    let n_features = matrix.ncols;
    let n_clusters = (n_samples as f64).sqrt().ceil() as usize;
    
    // Convert Matrix to Array2<f64>
    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = matrix.data[i][j] as f64;
        }
    }
    
    // Get initial random centroids
    let mut rng = rand::thread_rng();
    let indices = rand::seq::index::sample(&mut rng, n_samples, n_clusters).into_vec();
    let mut centroids = data.select(Axis(0), &indices);
    
    let tolerance = 1e-3;
    
    loop {
        // Assignment step: find closest centroid for each point
        let cluster_memberships = data.axis_iter(Axis(0))
            .map(|sample| find_closest_centroid(&centroids, &sample))
            .collect::<Array1<usize>>();
        
        // Update step: compute new centroids
        let new_centroids = compute_centroids(&data, &cluster_memberships, n_clusters);
        
        // Check convergence
        let distance = (&centroids - &new_centroids).mapv(|x| x * x).sum().sqrt();
        let has_converged = distance < tolerance;
        
        centroids = new_centroids;
        
        if has_converged {
            break;
        }
    }
    
    // Convert centroids back to Vec<Vec<u64>>
    Ok(centroids
        .axis_iter(Axis(0))
        .map(|row| row.iter().map(|&x| x.round() as u64).collect())
        .collect())
}

fn find_closest_centroid(centroids: &Array2<f64>, sample: &ArrayView1<f64>) -> usize {
    let mut closest_index = 0;
    let mut min_distance = f64::INFINITY;
    
    for (i, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
        let diff = &centroid - sample;
        let distance = diff.dot(&diff);
        if distance < min_distance {
            min_distance = distance;
            closest_index = i;
        }
    }
    
    closest_index
}

fn compute_centroids(data: &Array2<f64>, cluster_memberships: &Array1<usize>, n_clusters: usize) -> Array2<f64> {
    let (_, n_features) = data.dim();
    let mut centroids: HashMap<usize, RollingMean> = HashMap::new();
    
    for (sample, &cluster_index) in data.axis_iter(Axis(0)).zip(cluster_memberships.iter()) {
        if let Some(rolling_mean) = centroids.get_mut(&cluster_index) {
            rolling_mean.accumulate(&sample.to_owned());
        } else {
            let new_centroid = RollingMean::new(sample.to_owned());
            centroids.insert(cluster_index, new_centroid);
        }
    }
    
    let mut new_centroids = Array2::zeros((n_clusters, n_features));
    for (cluster_index, centroid) in centroids {
        new_centroids
            .row_mut(cluster_index)
            .assign(&centroid.current_mean);
    }
    
    new_centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clustering() -> Result<()> {
        let test_data = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];
        let matrix = Matrix { 
            data: test_data.clone(),
            nrows: 4,
            ncols: 4,
        };
        let centroids = get_centroids(&matrix)?;
        
        println!("Input data: {:?}", test_data);
        println!("Centroids: {:?}", centroids);
        
        assert!(centroids.iter().any(|c| c.iter().any(|&x| x > 0)));
        assert_eq!(centroids.len(), 2); // sqrt(4) = 2 clusters
        assert_eq!(centroids[0].len(), 4); // 4 features

        Ok(())
    }
} 