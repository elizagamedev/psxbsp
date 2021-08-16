//! This module triangulates a BSP tree's polygons into triangle strips.

use earcutr::earcut;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashMap;

use crate::{
    bsp::{BasicBspNode, BasicBspPolygon, BasicBspTree},
    geometry::{OrthonormalBasis2D, Plane, Vertex},
};

pub struct TriangulatedBspTree {
    pub nodes: Vec<TriangulatedBspNode>,
    pub texture_names: Vec<String>,
}

impl TriangulatedBspTree {
    pub fn from_basic_bsp_tree(tree: BasicBspTree) -> Self {
        let nodes = tree
            .nodes
            .into_iter()
            .map(|node| TriangulatedBspNode::from_basic_bsp_node(node))
            .collect();
        TriangulatedBspTree {
            nodes,
            texture_names: tree.texture_names,
        }
    }
}

pub struct TriangulatedBspNode {
    pub triangle_sets: Vec<TriangulatedBspTriangleSet>,
    pub plane: Plane,
    pub front_child: Option<usize>,
    pub back_child: Option<usize>,
}

impl TriangulatedBspNode {
    fn from_basic_bsp_node(node: BasicBspNode) -> Self {
        let basis = node.basis;
        let triangle_sets = collapse_triangle_sets(
            node.polygons
                .into_iter()
                .map(|x| TriangulatedBspTriangleSet::from_basic_bsp_polygon(&basis, x)),
        );
        Self {
            triangle_sets,
            plane: node.plane,
            front_child: node.front_child,
            back_child: node.back_child,
        }
    }
}

pub struct TriangulatedBspTriangleSet {
    pub indices: Vec<usize>,
    pub vertices: Vec<Vertex>,
    pub texture_ix: usize,
}

impl TriangulatedBspTriangleSet {
    fn from_basic_bsp_polygon(basis: &OrthonormalBasis2D, polygon: BasicBspPolygon) -> Self {
        let polygon2d: Vec<_> = polygon
            .vertices
            .iter()
            .map(|x| basis.transform(&x.position))
            .flatten()
            .collect();
        let mut indices = earcut(&polygon2d, &vec![], 2);
        indices.reverse();
        Self {
            indices,
            vertices: polygon.vertices,
            texture_ix: polygon.texture_ix,
        }
    }
}

/// Collapse triangle sets with the same texture index.
fn collapse_triangle_sets(
    triangle_sets: impl Iterator<Item = TriangulatedBspTriangleSet>,
) -> Vec<TriangulatedBspTriangleSet> {
    let mut map: HashMap<usize, TriangulatedBspTriangleSet> = HashMap::new();
    for triangle_set in triangle_sets {
        match map.entry(triangle_set.texture_ix) {
            Occupied(mut entry) => {
                let sum_set = entry.get_mut();
                let offset = sum_set.vertices.len();
                sum_set
                    .indices
                    .extend(triangle_set.indices.iter().map(|&ix| ix + offset));
                sum_set.vertices.extend(triangle_set.vertices);
            }
            Vacant(entry) => {
                entry.insert(triangle_set);
            }
        }
    }
    map.into_iter().map(|(_, v)| v).collect()
}
