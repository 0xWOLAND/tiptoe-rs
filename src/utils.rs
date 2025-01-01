use anyhow::Result;
use crate::{embeddings::TextEmbedder, SCALE_FACTOR};
use simplepir::Matrix;
use ndarray::{Array2, Axis, ArrayBase};
use std::ops::AddAssign;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn strings_to_embedding_matrix(texts: &[String], embedder: &TextEmbedder) -> Result<Matrix> {
    if texts.is_empty() {
        anyhow::bail!("No texts provided for embedding");
    }

    let mut embeddings_array = Array2::zeros((texts.len(), 384));
    
    for (i, text) in texts.iter().enumerate() {
        let embedding = embedder.embed(text)?;
        let embedding_vec = embedding.to_vec2::<f32>()?[0].clone();
        embeddings_array.row_mut(i).assign(&ndarray::ArrayView1::from(&embedding_vec));
    }

    let mut final_array = Array2::zeros((384, 384));
    final_array
        .slice_mut(ndarray::s![..embeddings_array.ncols(), ..texts.len()])
        .assign(&embeddings_array.t());

    let matrix_data: Vec<Vec<u64>> = final_array
        .axis_iter(Axis(0))
        .map(|row| {
            row.iter()
               .map(|&x| {
                    (x * SCALE_FACTOR).round() as u64
               })
               .collect()
        })
        .collect();

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
