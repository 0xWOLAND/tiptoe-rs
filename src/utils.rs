use anyhow::Result;
use nalgebra::{DMatrix, DVector};

fn encode_input(text: &str) -> Result<DVector<u64>> {
    println!("text: {}", text);
    let bytes = text.as_bytes();
    let tmp = bytes
        .chunks(8)
        .map(|chunk| {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(buf)
        })
        .collect::<Vec<u64>>();

    Ok(DVector::from_vec(tmp))
}


fn decode_input(data: &DVector<u64>) -> Result<String> {
    let bytes = data.iter().flat_map(|&x|{
        x.to_le_bytes().to_vec()
    }).collect::<Vec<u8>>();

    Ok(String::from_utf8(bytes)?)
}

pub fn encode_data(data: &Vec<String>) -> Result<DMatrix<u64>> {
    // First find the maximum length
    let max_length = data.iter().map(|text| text.len()).max().unwrap();
    
    // Pad each string to the same length
    let padded_data = data.iter()
        .map(|text| {
            let mut padded = text.clone();
            while padded.len() < max_length {
                padded.push('\0'); // Padding with null character
            }
            padded
        })
        .collect::<Vec<String>>();

    // Now encode the padded strings
    let embeddings = padded_data.iter()
        .map(|text| {
            println!("text: {}", text);
            encode_input(text).unwrap()
        })
        .collect::<Vec<_>>();
    
    let num_embeddings = embeddings.len();
    let embedding_size = embeddings[0].len();
    let square_size = std::cmp::max(num_embeddings, embedding_size);

    // Create a square matrix filled with zeros
    let mut square_matrix = DMatrix::zeros(square_size, square_size);
    
    // Copy the embeddings into the square matrix
    for (i, embedding) in embeddings.iter().enumerate() {
        for (j, &value) in embedding.iter().enumerate() {
            square_matrix[(i, j)] = value;
        }
    }
    
    Ok(square_matrix)
}

pub fn decode_data(data: &DMatrix<u64>) -> Result<Vec<String>> {
    // Decode and trim null characters
    let data = data.column_iter()
        .map(|col| {
            decode_input(&col.into_owned())
                .map(|s| s.trim_end_matches('\0').to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_encode_decode() {
        let data = vec!["Hello bitches!".to_string(), "world!".to_string()];
        let encoded = encode_data(&data).unwrap();
        let decoded = decode_data(&encoded).unwrap();
        println!("{:?}", decoded);
    }
}