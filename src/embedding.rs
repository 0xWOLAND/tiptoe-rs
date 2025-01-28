use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde_json::Value;
use tokenizers::Tokenizer;
use nalgebra::{DMatrix, DVector};

pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl BertEmbedder {
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let revision = "refs/pr/21".to_string();

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? 
        };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn normalize_l2(&self, v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }

    fn quantize_to_u64(&self, embeddings: &Tensor) -> Result<DVector<u64>> {
        let embeddings = embeddings.squeeze(0)?;
        let values = embeddings.to_vec1::<f32>()?;
        
        let max_value = 1 << 17;  // 131072
        let quantized: Vec<u64> = values.iter()
            .map(|&x| {
                let shifted = (x + 1.0) / 2.0;  // Shifts from [-1,1] to [0,1]
                (shifted * max_value as f32) as u64  // Scale to [0,131072]
            })
            .collect();
            
        Ok(DVector::from_vec(quantized))
    }

    pub fn embed_json_array(&self, json: &Vec<Value>) -> Result<DMatrix<u64>> {
        let embeddings = json.iter().map(|v| self.encode_text(&v.to_string())).collect::<Result<Vec<_>>>()?;

        let dim = std::cmp::max(embeddings[0].nrows(), embeddings.len());
        let mut out = DMatrix::zeros(dim, dim);

        for (i, embedding) in embeddings.iter().enumerate() {
            out.row_mut(i).copy_from_slice(embedding.as_slice());
        }

        Ok(out)
    }

    pub fn encode_text(&self, text: &str) -> Result<DVector<u64>> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let token_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        
        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;

        let embeddings = self.normalize_l2(&embeddings)?;
        
        self.quantize_to_u64(&embeddings)
    }

    pub fn compute_similarity(v1: &DVector<u64>, v2: &DVector<u64>) -> Result<f64> {
        if v1.nrows() != v2.nrows() {
            return Err(E::msg(format!(
                "Vector dimensions don't match: {} vs {}",
                v1.nrows(),
                v2.nrows()
            )));
        }

        let dot_product: f64 = v1
            .iter()
            .zip(v2.iter())
            .map(|(&x, &y)| {
                let x_norm = x as f64 / u64::MAX as f64;
                let y_norm = y as f64 / u64::MAX as f64;
                let x_shifted = 2.0 * x_norm - 1.0;
                let y_shifted = 2.0 * y_norm - 1.0;
                x_shifted * y_shifted
            })
            .sum();

        Ok(dot_product)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_shape() -> Result<()> {
        let embedder = BertEmbedder::new()?;
        let embedding = embedder.encode_text("test text")?;
        
        assert_eq!(embedding.nrows(), 384);
        Ok(())
    }

    #[test]
    fn test_quantization_range() -> Result<()> {
        let embedder = BertEmbedder::new()?;
        let embedding = embedder.encode_text("test text")?;
        
        assert!(embedding.iter().all(|&x| x <= u64::MAX));
        Ok(())
    }

    #[test]
    fn test_similarity() -> Result<()> {
        let embedder = BertEmbedder::new()?;
        
        let text1 = "The cat sits outside";
        let emb1 = embedder.encode_text(text1)?;
        let emb2 = embedder.encode_text(text1)?;
        let similarity = BertEmbedder::compute_similarity(&emb1, &emb2)?;
        println!("Identical text similarity: {}", similarity);
        assert!(similarity > 0.99, "Identical text similarity should be close to 1");

        let text3 = "Complex quantum physics theories";
        let emb3 = embedder.encode_text(text3)?;
        let dissimilar_score = BertEmbedder::compute_similarity(&emb1, &emb3)?;
        println!("Dissimilar text similarity: {}", dissimilar_score);
        
        assert!(similarity > dissimilar_score);
        assert!(dissimilar_score >= -1.0 && dissimilar_score <= 1.0);
        
        Ok(())
    }
}