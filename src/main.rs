use std::{collections::HashMap, time::Instant};

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

    fn search(&self, query: &str) -> Result<Vec<(Entry, f32)>> {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)
            .map_err(|err| anyhow::anyhow!(err))?;

        let encoding = tokenizer
            .encode(query, false)
            .map_err(|err| anyhow::anyhow!(err))?;

        let tokens = encoding.get_tokens();

        let mut result = Vec::<(u32, f32)>::new();
        for (id, _kv) in self.tf.clone() {
            let mut total_sum = 0f32;
            for token in tokens {
                let tf_rating = self.get_tf(&id, token);
                let idf_rating = self.get_idf(token);

                total_sum += tf_rating * idf_rating;
            }

            result.push((id, total_sum));
        }

        result.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let mut entries = Vec::new();
        for (id, score) in result {
            let entry = self.entries.iter().find(|e| e.id == id).unwrap();
            entries.push((entry.clone(), score));
        }

        Ok(entries)
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

    let now = Instant::now();
    db.tokenize_entries()
        .context("Failed to tokenize entries")?;
    println!("tokenize entries took {}ms", now.elapsed().as_millis());

    let term = "Green Smoothie";
    println!("search for {}", term);

    let now = Instant::now();
    let search_result = db.search(term)?;
    println!("search took {}ms", now.elapsed().as_millis());

    search_result
        .iter()
        .take(10)
        .for_each(|entry| println!("\t{} {} => {}", entry.0.id, entry.0.name, entry.1));

    Ok(())
}
