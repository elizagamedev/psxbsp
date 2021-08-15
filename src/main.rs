use anyhow::Result;
use obj::raw::parse_obj;
use std::{fs::File, io::BufReader};
use triangulation::TriangulatedBspTree;

mod bsp;
mod geometry;
mod reduce;
mod triangulation;

use crate::bsp::BasicBspTree;

fn main() -> Result<()> {
    let obj = parse_obj(BufReader::new(File::open("/home/eliza/bob.obj")?))?;
    let basic = BasicBspTree::from_wavefront_obj(obj)?;
    let triangulated = TriangulatedBspTree::from_basic_bsp_tree(basic);
    let reduced = ReducedBspTree::from_triangulated_bsp_tree(triangulated);

    Ok(())
}
