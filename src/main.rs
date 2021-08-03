use anyhow::Result;
use obj::raw::parse_obj;
use std::{fs::File, io::BufReader};

mod bsp;
mod dedup;
mod geometry;

use crate::bsp::BspTree;

fn main() -> Result<()> {
    let obj = parse_obj(BufReader::new(File::open("/home/eliza/bob.obj")?))?;
    let bsp = BspTree::new(obj)?;

    println!("{}", serde_yaml::to_string(&bsp).unwrap());

    Ok(())
}
