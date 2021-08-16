//! This module reduces a BSP tree's vertices into index-based lists.

use itertools::Itertools;
use nalgebra::{Vector2, Vector3};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::geometry::Plane;
use crate::triangulation::{TriangulatedBspNode, TriangulatedBspTree, TriangulatedBspTriangleSet};

pub struct ReducedBspTree {
    pub positions: Vec<Vector3<f64>>,
    pub nodes: Vec<ReducedBspNode>,
    pub texture_names: Vec<String>,
}

impl ReducedBspTree {
    pub fn from_triangulated_bsp_tree(tree: TriangulatedBspTree) -> Self {
        let mut position_index_map = PositionIndexMap::new();
        let nodes = tree
            .nodes
            .into_iter()
            .map(|node| ReducedBspNode::from_triangulated_bsp_node(node, &mut position_index_map))
            .collect();
        ReducedBspTree {
            positions: position_index_map.positions,
            nodes,
            texture_names: tree.texture_names,
        }
    }
}

pub struct ReducedBspNode {
    position_ixs: Vec<usize>,
    triangle_sets: Vec<ReducedBspTriangleSet>,
    plane: Plane,
    front_child: Option<usize>,
    back_child: Option<usize>,
}

impl ReducedBspNode {
    fn from_triangulated_bsp_node(
        node: TriangulatedBspNode,
        position_index_map: &mut PositionIndexMap,
    ) -> Self {
        let triangle_sets: Vec<ReducedBspTriangleSet> = node
            .triangle_sets
            .into_iter()
            .map(|set| {
                ReducedBspTriangleSet::from_triangulated_triangle_set(set, position_index_map)
            })
            .collect();
        let position_ixs = triangle_sets
            .iter()
            .map(|x| x.position_ixs.iter().copied().sorted())
            .kmerge()
            .dedup()
            .collect();
        Self {
            position_ixs,
            triangle_sets,
            plane: node.plane,
            front_child: node.front_child,
            back_child: node.back_child,
        }
    }
}

struct ReducedBspTriangleSet {
    position_ixs: Vec<usize>,
    tex_coords: Vec<Vector2<f64>>,
    texture_ix: usize,
}

impl ReducedBspTriangleSet {
    fn from_triangulated_triangle_set(
        triangle_set: TriangulatedBspTriangleSet,
        position_index_map: &mut PositionIndexMap,
    ) -> Self {
        let position_ixs = triangle_set
            .vertices
            .iter()
            .map(|x| position_index_map.get_ix(x.position))
            .collect();
        let tex_coords = triangle_set.vertices.iter().map(|x| x.tex_coord).collect();
        Self {
            position_ixs,
            tex_coords,
            texture_ix: triangle_set.texture_ix,
        }
    }
}

struct PositionIndexMap {
    map: HashMap<HashableVector3, usize>,
    positions: Vec<Vector3<f64>>,
}

impl PositionIndexMap {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            positions: Vec::new(),
        }
    }

    fn get_ix(&mut self, position: Vector3<f64>) -> usize {
        match self.map.entry(HashableVector3(position)) {
            Occupied(e) => *e.get(),
            Vacant(e) => {
                let result = self.positions.len();
                e.insert(result);
                self.positions.push(position);
                result
            }
        }
    }
}

struct HashableVector3(Vector3<f64>);

impl PartialEq for HashableVector3 {
    fn eq(&self, other: &Self) -> bool {
        return *self.0 == *other.0;
    }
}
impl Eq for HashableVector3 {}

impl Hash for HashableVector3 {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.0.x.to_bits().hash(state);
        self.0.y.to_bits().hash(state);
        self.0.z.to_bits().hash(state);
    }
}
