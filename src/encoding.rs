use anyhow::Result;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use simplepir::Matrix;
use std::io::{Read, Write};

#[derive(Debug)]
pub struct EncodedString(Vec<u64>);

impl From<&str> for EncodedString {
    fn from(s: &str) -> Self {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(s.as_bytes()).unwrap();
        Self(encoder.finish().unwrap().into_iter().map(|b| b as u64).collect())
    }
}

impl From<EncodedString> for String {
    fn from(encoded: EncodedString) -> Self {
        let bytes: Vec<u8> = encoded.0.iter().map(|&n| n as u8).collect();
        let mut decoder = ZlibDecoder::new(&bytes[..]);
        let mut s = String::new();
        decoder.read_to_string(&mut s).unwrap();
        s
    }
}

pub struct StringMatrix {
    data: Matrix,
    num_strings: usize,
}

impl StringMatrix {
    pub fn new(strings: &[String]) -> Self {
        let encoded: Vec<EncodedString> = strings.iter().map(|s| s.as_str().into()).collect();
        let max_len = encoded.iter().map(|e| e.0.len()).max().unwrap_or(0);
        let max_width = max_len + 1;
        let matrix_size = (strings.len() * max_width).max(max_width);
        
        let mut matrix_data = vec![vec![0u64; matrix_size]; matrix_size];
        for (i, nums) in encoded.iter().enumerate() {
            matrix_data[i][0] = nums.0.len() as u64;
            for (j, &num) in nums.0.iter().enumerate() {
                matrix_data[i][j + 1] = num;
            }
        }
        
        Self { 
            data: Matrix::from_data(matrix_data),
            num_strings: strings.len(),
        }
    }
}

impl From<StringMatrix> for Vec<String> {
    fn from(matrix: StringMatrix) -> Self {
        let mut strings = Vec::with_capacity(matrix.num_strings);
        for i in 0..matrix.num_strings {
            let length = matrix.data.data[i][0] as usize;
            if length > 0 {
                let encoded = EncodedString(matrix.data.data[i][1..=length].to_vec());
                strings.push(encoded.into());
            } else {
                strings.push(String::new());
            }
        }
        strings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_conversion() {
        let original = "Hello, World!";
        let encoded: EncodedString = original.into();
        let decoded: String = encoded.into();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_matrix_conversion() {
        let original = vec![
            "Hello, World!".to_string(),
            "This is a test".to_string(),
            "PIR is cool".to_string(),
        ];
        let matrix = StringMatrix::new(&original);
        let decoded: Vec<String> = matrix.into();
        assert_eq!(original, decoded);
    }
}
