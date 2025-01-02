use anyhow::Result;
use simplepir::Matrix;

#[derive(Debug)]
pub struct EncodedString(pub Vec<u64>);

impl From<&str> for EncodedString {
    fn from(s: &str) -> Self {
        let bytes = s.as_bytes();
        // Store length followed by packed bytes
        let mut data = vec![bytes.len() as u64];
        
        // Pack 8 bytes into each u64 (since 8 * 8 = 64 bits)
        for chunk in bytes.chunks(8) {
            let mut packed = 0u64;
            for (i, &byte) in chunk.iter().enumerate() {
                packed |= (byte as u64) << (i * 8);
            }
            data.push(packed);
        }
        Self(data)
    }
}

impl From<EncodedString> for String {
    fn from(encoded: EncodedString) -> Self {
        if encoded.0.is_empty() {
            return String::new();
        }
        
        let len = encoded.0[0] as usize;
        let mut bytes = Vec::with_capacity(len);
        
        // Unpack bytes from each u64
        for &packed in encoded.0.iter().skip(1) {
            for i in 0..8 {
                if bytes.len() >= len {
                    break;
                }
                let byte = ((packed >> (i * 8)) & 0xFF) as u8;
                bytes.push(byte);
            }
        }
        
        String::from_utf8(bytes).unwrap_or_default()
    }
}

pub struct StringMatrix {
    pub data: Matrix,
    num_strings: usize,
} 

impl StringMatrix {
    pub fn new(strings: &[String]) -> Self {
        let encoded: Vec<EncodedString> = strings.iter().map(|s| s.as_str().into()).collect();
        // Calculate max width needed (length + packed bytes)
        let max_len = encoded.iter()
            .map(|e| e.0.len())
            .max()
            .unwrap_or(0);
        let matrix_size = strings.len().max(max_len);
        
        let mut matrix_data = vec![vec![0u64; matrix_size]; matrix_size];
        for (i, nums) in encoded.iter().enumerate() {
            for (j, &num) in nums.0.iter().enumerate() {
                matrix_data[j][i] = num;
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
            let len = matrix.data.data[0][i] as usize;
            if len > 0 {
                // Calculate how many u64s we need based on length
                let packed_size = (len + 7) / 8;
                let encoded = EncodedString(
                    matrix.data.data[..=packed_size]
                        .iter()
                        .map(|row| row[i])
                        .collect()
                );
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

    #[test]
    fn test_long_string() {
        let original = "This is a much longer string that needs multiple u64s to store all of its bytes efficiently. Let's make sure it works correctly!";
        let encoded: EncodedString = original.into();
        let decoded: String = encoded.into();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_empty_string() {
        let original = "";
        let encoded: EncodedString = original.into();
        let decoded: String = encoded.into();
        assert_eq!(original, decoded);
    }
}
