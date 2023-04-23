use anyhow::{Context, Result};

mod search;
mod tokenizer;

fn main() -> Result<()> {
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

    let term = "Green Smoothie";

    let now = std::time::Instant::now();
    db.search(term)?
        .iter()
        .take(10)
        .for_each(|entry| println!("\t{} => {}", entry.0, entry.1));
    println!("Search took {}ms", now.elapsed().as_millis());

    Ok(())
}
