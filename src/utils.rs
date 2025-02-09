use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use num_traits::ops::bytes::ToBytes;

fn encode_input(text: &str) -> Result<DVector<u64>> {
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

pub fn decode_input(data: &DVector<BigInt>) -> Result<String> {
    let bytes = data
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<u8>>();

    let s = String::from_utf8(bytes)?;
    Ok(s.replace("\0", "")) // Remove ALL null characters
}

pub fn encode_data(data: &[String]) -> Result<DMatrix<BigInt>> {
    let max_length = data.iter().map(|text| text.len()).max().unwrap();

    let padded_data = data
        .iter()
        .map(|text| {
            let mut padded = text.clone();
            while padded.len() < max_length {
                padded.push('\0');
            }
            padded
        })
        .collect::<Vec<String>>();

    let embeddings = padded_data
        .iter()
        .map(|text| encode_input(text).unwrap())
        .collect::<Vec<_>>();

    let num_embeddings = embeddings.len();
    let embedding_size = embeddings[0].len();
    let square_size = std::cmp::max(num_embeddings, embedding_size);

    let mut square_matrix = DMatrix::zeros(square_size, square_size);

    for (i, embedding) in embeddings.iter().enumerate() {
        for (j, &value) in embedding.iter().enumerate() {
            square_matrix[(j, i)] = BigInt::from(value);
        }
    }

    Ok(square_matrix)
}

#[allow(dead_code)]
pub fn decode_data(data: &DMatrix<BigInt>) -> Result<Vec<String>> {
    let data = data
        .row_iter()
        .map(|row| {
            decode_input(&row.transpose().into_owned())
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
        let data = [
            "Lorem ipsum odor amet, consectetuer adipiscing elit",
            " Conubia elementum taciti dapibus vestibulum mattis primis",
            " Facilisis fames justo ultricies pharetra rhoncus",
            " Nam vel mi aptent turpis purus fusce purus",
            " Pretium ultrices torquent vulputate venenatis magnis vitae tempor semper torquent",
            " Habitant suspendisse nascetur in quis adipiscing",
        ]
        .map(|x| x.to_string())
        .to_vec();
        let encoded = encode_data(&data).unwrap();
        println!("encoded: {:?}", encoded);
        let decoded = decode_data(&encoded).unwrap();
        println!("{:?}", decoded);
    }
}
