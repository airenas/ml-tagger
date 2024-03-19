use serde::Serialize;
use warp::Rejection;

pub struct Service {}

pub type Result<T> = std::result::Result<T, Rejection>;

#[derive(Debug, Serialize, Clone)]
pub struct LiveResponse {
    pub status: bool,
}

#[derive(Debug, Serialize)]
pub struct Word {
    pub w: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub mi: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub lemma: String,
    #[serde(rename = "type")]
    pub w_type: String,
}
