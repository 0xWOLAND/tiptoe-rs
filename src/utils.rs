use anyhow::Result;
use crate::{embeddings::TextEmbedder, SCALE_FACTOR};
use simplepir::Matrix;
use ndarray::{Array2, Axis};

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
