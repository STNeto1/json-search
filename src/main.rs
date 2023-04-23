use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Deserialize;
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

#[derive(Debug, Clone)]
struct Entry {
    id: u32,
    name: String,
    author: String,
    user_rating: f64,
    reviews: u32,
    price: f64,
    year: u32,
}

impl Entry {
    fn get_tokens(&self) -> Vec<String> {
        vec![
            self.id.to_string(),
            self.name.clone(),
            self.author.clone(),
            self.user_rating.to_string(),
            self.reviews.to_string(),
            self.price.to_string(),
            self.year.to_string(),
        ]
    }
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

#[derive(Debug, Default)]
struct Database {
    pub entries: Vec<Entry>,
    tokens: HashMap<u32, Vec<String>>,

    tf: HashMap<u32, HashMap<String, u32>>,
}

impl Database {
    fn add(&mut self, entry: Entry) {
        let id = self.entries.len() as u32 + 1;

        self.entries.push(Entry { id, ..entry });
    }

    fn tokenize_entries(&mut self) -> Result<()> {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)
            .map_err(|err| anyhow::anyhow!(err))?;

        for entry in &self.entries {
            let encoding = tokenizer
                .encode(entry.get_tokens(), false)
                .map_err(|err| anyhow::anyhow!(err))?;

            let tokens = encoding.get_tokens();

            self.tokens.insert(entry.id, tokens.to_vec());

            let mut tf = HashMap::new();
            for token in tokens {
                let count = tf.entry(token.to_string()).or_insert(0);
                *count += 1;
            }
            self.tf.insert(entry.id, tf);
        }

        Ok(())
    }
}

fn main() -> Result<()> {
    let mut db = Database::default();

    // read file from assets
    csv::Reader::from_path("assets/data.csv")
        .context("Failed to read file")?
        .deserialize()
        .for_each(|record| {
            let mapped_record: Record = record.unwrap();
            let entry = Entry::from(mapped_record);
            db.add(entry);
        });

    db.tokenize_entries()
        .context("Failed to tokenize entries")?;

    println!("finished tokenizing {} entries", db.entries.len());

    Ok(())
}
