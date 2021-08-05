//! This module triangulates a BSP tree's polygons into triangle strips.

use earcutr::earcut;
use serde::Serialize;

use crate::{
    bsp::{BasicBspNode, BasicBspPolygon, BasicBspTree},
    geometry::{OrthonormalBasis2D, Plane, Vertex},
};

#[derive(Serialize)]
pub struct TriangulatedBspTree {
    pub nodes: Vec<Node>,
    pub texture_names: Vec<String>,
}

impl TriangulatedBspTree {
    pub fn from_basic_bsp_tree(tree: BasicBspTree) -> Self {
        let nodes = tree
            .nodes
            .into_iter()
            .map(|node| Node::from_basic_bsp_node(node))
            .collect();
        TriangulatedBspTree {
            nodes,
            texture_names: tree.texture_names,
        }
    }
}

#[derive(Serialize)]
pub struct Node {
    triangle_sets: Vec<TriangleSet>,
    plane: Plane,
    front_child: Option<usize>,
    back_child: Option<usize>,
}

impl Node {
    fn from_basic_bsp_node(node: BasicBspNode) -> Self {
        let basis = node.basis;
        let triangle_sets = node
            .polygons
            .into_iter()
            .map(|x| TriangleSet::from_basic_bsp_polygon(&basis, x))
            .collect();
        // TODO: collapse triangle sets with the same texture index.
        Self {
            triangle_sets,
            plane: node.plane,
            front_child: node.front_child,
            back_child: node.back_child,
        }
    }
}

#[derive(Serialize)]
struct TriangleSet {
    indices: Vec<usize>,
    vertices: Vec<Vertex>,
    texture_ix: usize,
}

impl TriangleSet {
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
