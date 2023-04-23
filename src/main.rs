use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use axum::{
    extract::{Query, State},
    routing, Json, Router,
};
use serde::{Deserialize, Serialize};

mod search;
mod tokenizer;

#[derive(Debug)]
struct AppState {
    db: search::Database,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut db = search::Database::new();

    // read file from assets
    csv::Reader::from_path("assets/data.csv")
        .context("Failed to read file")?
        .deserialize()
        .for_each(|record| {
            let mapped_record: search::Record = record.unwrap();
            let mut entry = search::Entry::from(mapped_record);
            entry.id = db.entries.len() as u32 + 1;
            let json_val = serde_json::to_value(entry).unwrap();

            db.add(json_val);
        });

    db.tokenize_entries()
        .context("Failed to tokenize entries")?;

    let app_state = Arc::new(Mutex::new(AppState { db }));

    let app = Router::new()
        .route("/search", routing::get(handle_search))
        .with_state(app_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();

    Ok(())
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    hits: Vec<serde_json::Value>,
    total_hits: u32,
    query: String,
    took: u128,
}

#[derive(Debug, Deserialize, Serialize)]
struct SearchQuery {
    query: Option<String>,
}

// This implementation is borderline bad practice, but it's just for the sake of the example
async fn handle_search(
    qs: Query<SearchQuery>,
    State(app_state): State<Arc<Mutex<AppState>>>,
) -> Json<SearchResponse> {
    let now = std::time::Instant::now();

    let query_term = qs.query.clone().unwrap_or_default();

    let search_result = app_state
        .lock()
        .unwrap()
        .db
        .search(query_term.as_str())
        .unwrap();

    let response = SearchResponse {
        hits: search_result.iter().take(10).map(|r| r.0.clone()).collect(),
        total_hits: search_result.len() as u32,
        query: query_term,
        took: now.elapsed().as_millis(),
    };

    Json(response)
}
