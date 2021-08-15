//! This module reduces a BSP tree's vertices into index-based lists.

use earcutr::earcut;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashMap;

use crate::geometry::{OrthonormalBasis2D, Plane, Vertex};
use crate::triangulation::{TriangulatedBspNode, TriangulatedBspTree};

pub struct ReducedBspTree {
    pub nodes: Vec<ReducedBspNode>,
    pub texture_names: Vec<String>,
}

impl ReducedBspTree {
    pub fn from_triangulated_bsp_tree(tree: TriangulatedBspTree) -> Self {
        let nodes = tree
            .nodes
            .into_iter()
            .map(|node| ReducedBspNode::from_triangulated_bsp_node(node))
            .collect();
        ReducedBspTree {
            nodes,
            texture_names: tree.texture_names,
        }
    }
}

pub struct ReducedBspNode {
    triangle_sets: Vec<ReducedBspTriangleSet>,
    plane: Plane,
    front_child: Option<usize>,
    back_child: Option<usize>,
}

impl ReducedBspNode {
    fn from_triangulated_bsp_node(node: TriangulatedBspNode) -> Self {
        let basis = node.basis;
        let triangle_sets = collapse_triangle_sets(
            node.polygons
                .into_iter()
                .map(|x| ReducedBspTriangleSet::from_basic_bsp_polygon(&basis, x)),
        );
        Self {
            triangle_sets,
            plane: node.plane,
            front_child: node.front_child,
            back_child: node.back_child,
        }
    }
}

struct ReducedBspTriangleSet {
    indices: Vec<usize>,
    texture_ix: usize,
}

impl ReducedBspTriangleSet {
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
    triangle_sets: impl Iterator<Item = ReducedBspTriangleSet>,
) -> Vec<ReducedBspTriangleSet> {
    let mut map: HashMap<usize, ReducedBspTriangleSet> = HashMap::new();
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
