use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::tokenizer;

#[derive(Debug, Deserialize)]
pub struct Record {
    #[serde(rename = "Name")]
    name: String,

    #[serde(rename = "Author")]
    author: String,

    #[serde(rename = "User Rating")]
    user_rating: f64,

    #[serde(rename = "Reviews")]
    reviews: u32,

    #[serde(rename = "Price")]
    price: f64,

    #[serde(rename = "Year")]
    year: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub id: u32,
    pub name: String,
    pub author: String,
    pub user_rating: f64,
    pub reviews: u32,
    pub price: f64,
    pub year: u32,
}

impl From<Record> for Entry {
    fn from(record: Record) -> Self {
        Self {
            id: 0,
            name: record.name,
            author: record.author,
            user_rating: record.user_rating,
            reviews: record.reviews,
            price: record.price,
            year: record.year,
        }
    }
}

#[derive(Debug)]
struct CachedResult {
    result: Vec<(serde_json::Value, f32)>,
    expiration: std::time::Instant,
}

#[derive(Debug)]
pub struct Database {
    pub entries: Vec<serde_json::Value>,
    tokens: HashMap<u32, Vec<String>>,

    tf: HashMap<u32, HashMap<String, u32>>,
    // tokenizer: Tokenizer,
    cached_results: HashMap<String, CachedResult>,
}

impl Database {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            tokens: HashMap::new(),
            tf: HashMap::new(),
            cached_results: HashMap::new(),
        }
    }

    pub fn add(&mut self, entry: serde_json::Value) {
        self.entries.push(entry);
    }

    fn get_tokens(&self, val: &serde_json::Value) -> Option<Vec<String>> {
        return match val {
            serde_json::Value::Object(map) => {
                let mut tokens = Vec::new();
                for (_, value) in map {
                    match &mut self.get_tokens(value) {
                        Some(v) => tokens.append(v),
                        None => {}
                    }
                }
                Some(tokens)
            }
            serde_json::Value::Array(arr) => {
                let mut tokens = Vec::new();
                for value in arr {
                    match &mut self.get_tokens(value) {
                        Some(v) => tokens.append(v),
                        None => {}
                    }
                }
                Some(tokens)
            }
            serde_json::Value::String(s) => Some(vec![s.to_string()]),
            serde_json::Value::Number(n) => Some(vec![n.to_string()]),
            serde_json::Value::Bool(b) => Some(vec![b.to_string()]),
            serde_json::Value::Null => None,
        };
    }

    pub fn tokenize_entries(&mut self) -> Result<()> {
        for entry in &self.entries {
            let entry_tokens = self
                .get_tokens(entry)
                .context("Failed to get tokens")?
                .iter()
                .map(|x| {
                    tokenizer::Lexer::new(x.to_owned().chars().collect::<Vec<_>>().as_slice())
                        .into_iter()
                        .map(|y| y.to_ascii_uppercase())
                        .collect::<Vec<String>>()
                })
                .flatten()
                .collect::<Vec<String>>();

            /*
                        let tokens = tokenizer::Lexer::new(query.chars().collect::<Vec<_>>().as_slice())
                            .into_iter()
                            .map(|x| x.to_ascii_uppercase())
                            .collect::<Vec<String>>();
            */

            let entry_id_key = entry.get("id").context("Failed to get id")?;
            let id = entry_id_key
                .as_u64()
                .context("Failed to convert id to u64")?;

            self.tokens.insert(id as u32, entry_tokens.clone());

            let mut tf = HashMap::new();
            for token in entry_tokens {
                let count = tf.entry(token.to_string()).or_insert(0);
                *count += 1;
            }
            self.tf.insert(id as u32, tf);
        }

        Ok(())
    }

    fn get_tf(&self, key: &u32, token: &str) -> f32 {
        let top = self.tf.get(key).unwrap().get(token).clone().unwrap_or(&0);
        let bottom = self.tf.get(key).unwrap().values().sum::<u32>();

        return *top as f32 / bottom as f32;
    }

    fn get_idf(&self, token: &str) -> f32 {
        let n = self.entries.len() as f32;
        let df = self
            .tf
            .values()
            .filter(|kv| kv.contains_key(token))
            .count()
            .max(1) as f32;

        return (n / df).ln();
    }

    fn calculate_tf_idf(&mut self, key: &u32, token: &str) -> f32 {
        let tf = self.get_tf(key, token);
        let idf = self.get_idf(token);

        return tf * idf;
    }

    pub fn search(&mut self, query: &str) -> Result<Vec<(serde_json::Value, f32)>> {
        if let Some(cached_result) = self.cached_results.get(query) {
            if cached_result.expiration > std::time::Instant::now() {
                return Ok(cached_result.result.clone());
            }
        }

        let tokens = tokenizer::Lexer::new(query.chars().collect::<Vec<_>>().as_slice())
            .into_iter()
            .map(|x| x.to_ascii_uppercase())
            .collect::<Vec<String>>();

        let mut result = self
            .tf
            .clone()
            .keys()
            .into_iter()
            .map(|id| {
                (
                    *id,
                    tokens
                        .iter()
                        .map(|t| self.calculate_tf_idf(id, t))
                        .sum::<f32>(),
                )
            })
            .filter(|(_, score)| *score > 0.0)
            .collect::<Vec<(u32, f32)>>();
        result.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let mut entries = Vec::new();
        for (id, score) in result {
            let entry = self
                .entries
                .iter()
                .find(|e| e.get("id").unwrap().as_u64().unwrap() == id as u64)
                .unwrap()
                .clone();
            entries.push((entry, score));
        }

        self.cached_results.insert(
            query.to_string(),
            CachedResult {
                result: entries.clone(),
                expiration: std::time::Instant::now() + std::time::Duration::from_secs(60),
            },
        );

        Ok(entries)
    }
}
