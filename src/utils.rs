use anyhow::Result;
use crate::{embeddings::TextEmbedder, SCALE_FACTOR};
use simplepir::Matrix;
use ndarray::{Array2, Axis, ArrayBase};
use std::ops::AddAssign;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn strings_to_embedding_matrix(texts: &[String], embedder: &TextEmbedder) -> Result<Matrix> {
    let mut scaled_embeddings = Vec::with_capacity(texts.len());
    
    // First convert all embeddings to scaled u64s
    for text in texts {
        let embedding = embedder.embed(text)?;
        scaled_embeddings.push(scale_to_u64(embedding)?);
    }
    
    let matrix_size = 384.max(texts.len());
    let mut matrix_data = vec![vec![0u64; matrix_size]; matrix_size];
    
    for (col, embedding) in scaled_embeddings.iter().enumerate() {
        for (row, &value) in embedding.iter().take(384).enumerate() {
            matrix_data[row][col] = value;
        }
    }
    
    Ok(Matrix::from_data(matrix_data))
}

pub fn kmeans_cluster(data: &Matrix) -> Result<Vec<Vec<u64>>> {
    let n_samples = data.data[0].len();
    let n_features = data.data.len();
    let k = (n_samples as f32).sqrt().round() as usize;
    let max_iterations = 100;
    let tolerance = 1e-4;

    let data_array: Array2<f32> = Array2::from_shape_vec(
        (n_samples, n_features),
        (0..n_samples).flat_map(|i| data.data.iter().map(move |row| row[i] as f32)).collect()
    )?;

    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    let mut centroids = Array2::zeros((k, n_features));
    for (i, &idx) in indices.iter().take(k).enumerate() {
        centroids.row_mut(i).assign(&data_array.row(idx));
    }

    let mut old_centroids = Array2::zeros((k, n_features));
    let mut iterations = 0;

    while iterations < max_iterations {
        old_centroids.assign(&centroids);

        let mut cluster_assignments = vec![0; n_samples];
        let mut cluster_sizes = vec![0; k];

        for (i, sample) in data_array.outer_iter().enumerate() {
            let mut min_dist = f32::INFINITY;
            let mut closest_cluster = 0;

            for (j, centroid) in centroids.outer_iter().enumerate() {
                let dist: f32 = sample
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                if dist < min_dist {
                    min_dist = dist;
                    closest_cluster = j;
                }
            }

            cluster_assignments[i] = closest_cluster;
            cluster_sizes[closest_cluster] += 1;
        }

        centroids.fill(0.0);
        for (i, sample) in data_array.outer_iter().enumerate() {
            let cluster = cluster_assignments[i];
            centroids.row_mut(cluster).add_assign(&sample);
        }

        for (i, &size) in cluster_sizes.iter().enumerate() {
            if size > 0 {
                centroids.row_mut(i).mapv_inplace(|x| x / size as f32);
            }
        }

        let centroid_shift: f32 = centroids
            .iter()
            .zip(old_centroids.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        if centroid_shift < tolerance {
            break;
        }

        iterations += 1;
    }

    Ok(centroids
        .outer_iter()
        .map(|row| row.iter().map(|&x| x.round() as u64).collect())
        .collect())
}

pub fn scale_to_u64(tensor: candle_core::Tensor) -> Result<Vec<u64>> {
    Ok(tensor.to_vec2::<f32>()?
        .into_iter()
        .next()
        .unwrap()
        .into_iter()
        .map(|x| (x * SCALE_FACTOR).round() as u64)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_clustering() {
        let data = Matrix::from_data(vec![
            vec![1, 2, 8, 9, 3, 2],
            vec![2, 1, 9, 8, 2, 3],
            vec![1, 3, 7, 8, 2, 1],
            vec![8, 9, 2, 1, 7, 8],
            vec![9, 8, 1, 2, 8, 7],
            vec![7, 8, 3, 2, 9, 8],
            vec![2, 1, 8, 9, 2, 3],
            vec![1, 2, 9, 8, 3, 2],
        ]);

        let centroids = kmeans_cluster(&data).unwrap();
        assert_eq!(centroids[0].len(), 8);

        for row in &centroids {
            assert!(row.iter().all(|&x| x <= 9));
        }
    }
}
