use anyhow::{Context, Result};
use serde::Deserialize;

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

fn main() -> Result<()> {
    // read file from assets
    let result = csv::Reader::from_path("assets/data.csv")
        .context("Failed to read file")?
        .deserialize()
        .map(|record| {
            let record: Record = record.context("Failed to deserialize record")?;
            Ok(record)
        })
        .collect::<Result<Vec<_>>>()
        .context("Failed to process records")?;

    for record in result {
        println!("{:?}", record);
    }

    Ok(())
}
