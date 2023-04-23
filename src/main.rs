use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct Record {
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
struct Entry {
    id: u32,
    name: String,
    author: String,
    user_rating: f64,
    reviews: u32,
    price: f64,
    year: u32,
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
struct Database {
    pub entries: Vec<serde_json::Value>,
    tokens: HashMap<u32, Vec<String>>,

    tf: HashMap<u32, HashMap<String, u32>>,
    tokenizer: Tokenizer,
}

impl Database {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            tokens: HashMap::new(),
            tf: HashMap::new(),
            tokenizer: Tokenizer::from_pretrained("bert-base-cased", None)
                .map_err(|err| anyhow::anyhow!(err))
                .unwrap(),
        }
    }

    fn add(&mut self, entry: serde_json::Value) {
        self.entries.push(entry);
    }

    fn get_tokens(&self, val: &serde_json::Value) -> Option<Vec<String>> {
        // self.tokens.get(key).unwrap().clone()
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

    fn tokenize_entries(&mut self) -> Result<()> {
        for entry in &self.entries {
            let entry_tokens = self.get_tokens(entry).context("Failed to get tokens")?;

            let encoding = self
                .tokenizer
                .encode(entry_tokens, false)
                .map_err(|err| anyhow::anyhow!(err))?;

            let tokens = encoding
                .get_tokens()
                .iter()
                .map(|x| x.to_ascii_uppercase())
                .collect::<Vec<String>>();

            let entry_id_key = entry.get("id").context("Failed to get id")?;
            let id = entry_id_key
                .as_u64()
                .context("Failed to convert id to u64")?;

            self.tokens.insert(id as u32, tokens.to_vec());

            let mut tf = HashMap::new();
            for token in tokens {
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

    fn search(&mut self, query: &str) -> Result<Vec<(serde_json::Value, f32)>> {
        let encoding = self
            .tokenizer
            .encode(query, false)
            .map_err(|err| anyhow::anyhow!(err))?;

        let tokens: Vec<String> = encoding
            .get_tokens()
            .iter()
            .map(|x| x.to_ascii_uppercase())
            .collect();

        let mut result = self
            .tf
            .clone()
            .keys()
            .into_iter()
            .map(|id| {
                (
                    *id,
                    tokens.iter().map(|t| self.calculate_tf_idf(id, t)).sum(),
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

        Ok(entries)
    }
}

fn main() -> Result<()> {
    let mut db = Database::new();

    // read file from assets
    csv::Reader::from_path("assets/data.csv")
        .context("Failed to read file")?
        .deserialize()
        .for_each(|record| {
            let mapped_record: Record = record.unwrap();
            let mut entry = Entry::from(mapped_record);
            entry.id = db.entries.len() as u32 + 1;
            let json_val = serde_json::to_value(entry).unwrap();

            db.add(json_val);
        });

    db.tokenize_entries()
        .context("Failed to tokenize entries")?;

    let term = "Green Smoothie";

    let now = std::time::Instant::now();
    db.search(term)?
        .iter()
        .take(10)
        .for_each(|entry| println!("\t{} => {}", entry.0, entry.1));
    println!("Search took {}ms", now.elapsed().as_millis());

    Ok(())
}
