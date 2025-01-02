use anyhow::Result;
use simplepir::Matrix;

#[derive(Debug)]
pub struct EncodedString(pub Vec<u64>);
impl From<&str> for EncodedString {
    fn from(s: &str) -> Self {
        Self(s.as_bytes().iter().map(|&b| b as u64).collect())
    }
}

impl From<EncodedString> for String {
    fn from(encoded: EncodedString) -> Self {
        String::from_utf8(encoded.0.iter().map(|&n| n as u8).collect()).unwrap()
    }
}

pub struct StringMatrix {
    pub data: Matrix,
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
            matrix_data[0][i] = nums.0.len() as u64;
            for (j, &num) in nums.0.iter().enumerate() {
                matrix_data[j + 1][i] = num;
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
            let length = matrix.data.data[0][i] as usize;
            if length > 0 {
                let encoded = EncodedString(matrix.data.data[1..=length].iter().map(|row| row[i]).collect());
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
