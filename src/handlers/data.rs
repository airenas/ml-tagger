use serde::{Deserialize, Serialize};
use warp::Rejection;

pub struct WorkContext {
    pub params: TagParams,
    pub embeddings: Vec<f32>,
    pub sentences: Vec<Vec<WorkWord>>,
}

pub struct WorkWord {
    pub w: String,
    pub mi: Option<String>,
    pub lemma: Option<String>,
    pub w_type: Option<String>,
    pub embeddings: Option<Vec<f32>>,
    pub predicted: Option<i32>,
    pub predicted_str: Option<String>,
}

pub trait Processor {
    fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()>;
}

pub struct Service {
    pub embedder: Box<dyn Processor + Send + Sync>,
    pub onnx: Box<dyn Processor + Send + Sync>,
    pub tag_mapper: Box<dyn Processor + Send + Sync>,
    pub calls: u32,
}

pub type Result<T> = std::result::Result<T, Rejection>;

#[derive(Debug, Serialize, Clone)]
pub struct LiveResponse {
    pub status: bool,
}

#[derive(Deserialize)]
pub struct TagParams {
    pub debug: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct Word {
    pub w: String,
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
}

impl WorkContext {
    pub fn new(params: TagParams) -> WorkContext {
        WorkContext {
            params,
            embeddings: Vec::<f32>::new(),
            sentences: Vec::<Vec<WorkWord>>::new(),
        }
    }
}

impl WorkWord {
    pub fn new(w: String) -> WorkWord {
        WorkWord {
            w,
            mi: None,
            lemma: None,
            w_type: None,
            embeddings: None,
            predicted: None,
            predicted_str: None,
        }
    }
}
