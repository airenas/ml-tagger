use std::borrow::Cow;

use axum::response::IntoResponse;
use reqwest::StatusCode;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("bad request: {0}, details: {1}")]
    BadRequest(String, String),
    #[error("PProf not activated")]
    PProfNotActivated(),
    #[error("Server error: {0}")]
    Server(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message): (StatusCode, Cow<'static, str>) = match self {
            ApiError::BadRequest(msg, details) => {
                tracing::warn!("{}: {}", msg, details);
                (StatusCode::BAD_REQUEST, Cow::Owned(msg))
            }
            ApiError::Server(msg) => {
                tracing::error!("{}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Cow::Borrowed("Internal Server Error"),
                )
            }
            ApiError::Other(err) => {
                tracing::error!("{}", err);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Cow::Borrowed("Internal Server Error"),
                )
            }
            ApiError::PProfNotActivated() => {
                tracing::error!("PProf not activated");
                (
                    StatusCode::FORBIDDEN,
                    Cow::Borrowed("PProf not activated"),
                )
            }
        };

        (status, message).into_response()
    }
}
