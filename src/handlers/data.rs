use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::error::ApiError;

pub struct WorkContext {
    pub params: TagParams,
    pub embeddings: Vec<f32>,
    pub text: String,
    pub sentences: Vec<Vec<WorkWord>>,
}

#[derive(Clone)]
pub struct WorkMI {
    pub mi: Option<String>,
    pub lemma: Option<String>,
}

pub struct WorkWord {
    pub w: String,
    pub is_word: bool,
    pub mi: Option<String>,
    pub lemma: Option<String>,
    pub w_type: Option<String>,
    pub embeddings: Option<Arc<Vec<f32>>>,
    pub predicted: Option<i32>,
    pub predicted_str: Option<String>,
    pub mis: Option<Arc<Vec<WorkMI>>>,
}

#[async_trait]
pub trait Processor {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()>;
}

pub struct Service {
    pub lexer: Box<dyn Processor + Send + Sync>,
    pub embedder: Box<dyn Processor + Send + Sync>,
    pub onnx: Box<dyn Processor + Send + Sync>,
    pub tag_mapper: Box<dyn Processor + Send + Sync>,
    pub lemmatize_words_mapper: Box<dyn Processor + Send + Sync>,
    pub clitics: Box<dyn Processor + Send + Sync>,
    pub static_words: Box<dyn Processor + Send + Sync>,
    pub restorer: Box<dyn Processor + Send + Sync>,
    pub calls: u32,
}

pub type ApiResult<T> = std::result::Result<T, ApiError>;

#[derive(Debug, Serialize, Clone)]
pub struct LiveResponse {
    pub status: bool,
    pub version: String,
}

#[derive(Deserialize)]
pub struct TagParams {
    pub debug: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MI {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mi: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lemma: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Word {
    #[serde(rename = "string", skip_serializing_if = "Option::is_none")]
    pub w: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mi: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lemma: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub w_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_str: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mis: Option<Vec<MI>>,
}

impl WorkContext {
    pub fn new(params: TagParams, text: String) -> WorkContext {
        WorkContext {
            params,
            text,
            embeddings: Vec::<f32>::new(),
            sentences: Vec::<Vec<WorkWord>>::new(),
        }
    }
}

impl WorkWord {
    pub fn new(w: String, is_word: bool) -> WorkWord {
        WorkWord {
            w,
            is_word,
            mi: None,
            lemma: None,
            w_type: None,
            embeddings: None,
            predicted: None,
            predicted_str: None,
            mis: None,
        }
    }
}
