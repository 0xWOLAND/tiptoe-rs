use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct TextEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl TextEmbedder {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;

        let model_id = "sentence-transformers/all-MiniLM-L6-v2";
        let revision = "main";
        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());

        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = api.get("model.safetensors")?;

        let config = std::fs::read_to_string(&config_filename).map_err(anyhow::Error::msg)?;
        let config: Config = serde_json::from_str(&config).map_err(anyhow::Error::msg)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(anyhow::Error::msg)?;

        // Load model
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[&weights_filename], DType::F32, &device)
                .map_err(anyhow::Error::msg)? 
        };
        let model = BertModel::load(vb, &config).map_err(anyhow::Error::msg)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Tensor> {
        // Tokenize input
        let tokens = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        let token_ids = tokens.get_ids().to_vec();

        // Convert to tensors
        let token_ids = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        // Get embeddings
        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        
        // Average pooling over token dimension
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        
        Ok(embeddings)
    }

    pub fn cosine_similarity(&self, embedding1: &Tensor, embedding2: &Tensor) -> Result<f32> {
        // Flatten embeddings to 1D
        let embedding1 = embedding1.flatten_all()?;
        let embedding2 = embedding2.flatten_all()?;
        
        // Compute dot product
        let dot_product = (&embedding1 * &embedding2)?.sum_all()?;
        
        // Compute norms
        let norm1 = embedding1.sqr()?.sum_all()?.sqrt()?;
        let norm2 = embedding2.sqr()?.sum_all()?.sqrt()?;
        
        // Compute cosine similarity and convert to f32
        let similarity = (dot_product / (norm1 * norm2))?.to_vec0::<f32>()?;
        Ok(similarity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identical_string_similarity() -> Result<()> {
        let embedder = TextEmbedder::new()?;
        let text = "This is a test sentence.";
        
        let embedding1 = embedder.embed(text)?;
        let embedding2 = embedder.embed(text)?;
        
        let similarity = embedder.cosine_similarity(&embedding1, &embedding2)?;
        
        // Identical strings should have cosine similarity of 1.0
        assert_relative_eq!(similarity, 1.0, epsilon = 1e-5);
        
        Ok(())
    }

    #[test]
    fn test_similar_strings() -> Result<()> {
        let embedder = TextEmbedder::new()?;
        let text1 = "I love programming in Rust";
        let text2 = "I enjoy coding in Rust";
        
        let embedding1 = embedder.embed(text1)?;
        let embedding2 = embedder.embed(text2)?;
        
        let similarity = embedder.cosine_similarity(&embedding1, &embedding2)?;
        
        // Similar strings should have high but not perfect similarity
        assert!(similarity > 0.8, "Similar strings should have high similarity");
        assert!(similarity < 1.0, "Different strings should not have perfect similarity");
        
        Ok(())
    }

    #[test]
    fn test_different_strings() -> Result<()> {
        let embedder = TextEmbedder::new()?;
        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "Machine learning models are fascinating";
        
        let embedding1 = embedder.embed(text1)?;
        let embedding2 = embedder.embed(text2)?;
        
        let similarity = embedder.cosine_similarity(&embedding1, &embedding2)?;
        
        // Different topics should have moderate to low similarity
        assert!(similarity < 0.95, "Very different topics should not have very high similarity");
        assert!(similarity > -1.0, "Similarity should be greater than -1");
        println!("Similarity between different topics: {}", similarity);
        
        Ok(())
    }
}