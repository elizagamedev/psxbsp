use crate::geometry::{collapse_planes, Plane, PlaneSide, Polygon, Vertex};
use anyhow::{anyhow, Result};
use float_cmp::approx_eq;
use itertools::izip;
use nalgebra::{Vector2, Vector3};
use obj::raw::RawObj;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{HashMap, HashSet};

#[derive(Serialize, Debug)]
struct ClassifiedPolygons {
    plane: Plane,
    coplanar_ixs: HashSet<usize>,
    front_ixs: HashSet<usize>,
    back_ixs: HashSet<usize>,
    intersecting_ixs: HashMap<usize, Vec<PlaneSide>>,
}

impl ClassifiedPolygons {
    fn new(plane: Plane, coplanar_ixs: HashSet<usize>, polygons: &[Polygon]) -> ClassifiedPolygons {
        let mut result = ClassifiedPolygons {
            plane,
            coplanar_ixs,
            front_ixs: HashSet::new(),
            back_ixs: HashSet::new(),
            intersecting_ixs: HashMap::new(),
        };

        for (i, other) in polygons.iter().enumerate() {
            if !result.coplanar_ixs.contains(&i) {
                result.classify(i, other);
            }
        }

        result
    }

    /// Classify the given polygon as at least partially in front, at least
    /// partially behind, or intersecting our plane, adding its index to the
    /// corresponding sets.
    fn classify(&mut self, polygon_ix: usize, polygon: &Polygon) {
        let sides: Vec<PlaneSide> = polygon
            .vertices
            .iter()
            .map(|v| {
                let this_distance = v.position.dot(&self.plane.normal);
                if approx_eq!(
                    f64,
                    this_distance,
                    self.plane.distance,
                    epsilon = 0.01,
                    ulps = 2i64.pow(48)
                ) {
                    PlaneSide::Coplanar
                } else if this_distance > self.plane.distance {
                    PlaneSide::Front
                } else {
                    PlaneSide::Back
                }
            })
            .collect();
        if sides
            .iter()
            .all(|side| matches!(side, PlaneSide::Front | PlaneSide::Coplanar))
        {
            self.front_ixs.insert(polygon_ix);
        } else if sides
            .iter()
            .all(|side| matches!(side, PlaneSide::Back | PlaneSide::Coplanar))
        {
            self.back_ixs.insert(polygon_ix);
        } else {
            self.intersecting_ixs.insert(polygon_ix, sides);
            self.front_ixs.insert(polygon_ix);
            self.back_ixs.insert(polygon_ix);
        }
    }

    /// Given a polygon that has been split. We only need to recalculate the
    /// full reclassification if this polygon was previously classified as an
    /// intersection, since it may potentially now be wholly in front or behind.
    fn reclassify_split(&mut self, polygon_ix: usize, polygon: &Polygon) {
        match self.intersecting_ixs.entry(polygon_ix) {
            Occupied(entry) => entry.remove_entry(),
            Vacant(_) => return,
        };
        self.front_ixs.remove(&polygon_ix);
        self.back_ixs.remove(&polygon_ix);

        self.classify(polygon_ix, polygon);
    }

    /// Return a new plane that contains only the polygons specified in `other`.
    /// Returns `None` if our coplanar set is reduced to nothing.
    fn intersection(&self, other: &HashSet<usize>) -> Option<Self> {
        let coplanar_ixs: HashSet<_> = self.coplanar_ixs.intersection(other).copied().collect();
        if coplanar_ixs.is_empty() {
            return None;
        }
        let front_ixs = self.front_ixs.intersection(other).copied().collect();
        let back_ixs = self.back_ixs.intersection(other).copied().collect();
        let intersecting_ixs = self
            .intersecting_ixs
            .iter()
            .filter(|(k, _)| other.contains(k))
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        Some(ClassifiedPolygons {
            plane: self.plane.clone(),
            coplanar_ixs,
            front_ixs,
            back_ixs,
            intersecting_ixs,
        })
    }
}

#[derive(Serialize, Debug)]
struct Node {
    polygons: Vec<Polygon>,
    normal: Vector3<f64>,
    distance: f64,
    children: [Option<usize>; 2],
}

impl Node {
    fn new(
        polygons: Vec<Polygon>,
        normal: Vector3<f64>,
        distance: f64,
        children: [Option<usize>; 2],
    ) -> Self {
        Self {
            polygons,
            normal,
            distance,
            children,
        }
    }
}

#[derive(Serialize)]
pub struct BspTree {
    nodes: Vec<Node>,
    texture_names: Vec<String>,
}

impl BspTree {
    pub fn new(obj: RawObj) -> Result<Self> {
        let (polygons, texture_names) = collect_polygons(obj)?;

        let planes: Vec<Plane> = polygons
            .par_iter()
            .map(|polygon| polygon.to_plane())
            .collect();

        if izip!(&polygons, &planes).any(|(polygon, plane)| !polygon.is_planar(plane)) {
            return Err(anyhow!("Non-planar polygon encountered"));
        }

        let collapsed_planes = collapse_planes(&planes);

        let planes = collapsed_planes
            .into_par_iter()
            .map(|(plane, coplanar_ixs)| ClassifiedPolygons::new(plane, coplanar_ixs, &polygons))
            .collect();

        // println!("{}", serde_yaml::to_string(&planes).unwrap());

        let nodes = build_tree(polygons, planes);

        Ok(BspTree {
            nodes,
            texture_names,
        })
    }
}

fn collect_polygons(obj: obj::raw::RawObj) -> Result<(Vec<Polygon>, Vec<String>)> {
    // We expect a model with Up as +Y, Forward as -Z, and vertices arranged in
    // counter-clockwise order. We want a coordinate system with Up as -Y,
    // Forward as +Z, and vertices arranged in clockwise order. We achieve this
    // by negating the Y and Z coordinates and reversing the vertex order.
    let positions: Vec<_> = obj
        .positions
        .iter()
        .map(|x| Vector3::new(x.0 as f64, -x.1 as f64, -x.2 as f64))
        .collect();

    let tex_coords: Vec<_> = obj
        .tex_coords
        .iter()
        .map(|x| Vector2::new(x.0 as f64, x.1 as f64))
        .collect();

    let mut texture_names = Vec::with_capacity(obj.meshes.len());
    let mut polygons: Vec<_> = Vec::with_capacity(obj.polygons.len());

    for (texture_ix, (texture_name, group)) in obj.meshes.into_iter().enumerate() {
        texture_names.push(texture_name);
        for polygon_range in group.polygons {
            for polygon_ix in polygon_range.start..polygon_range.end {
                let vertices = match &obj.polygons[polygon_ix] {
                    obj::raw::object::Polygon::PT(ref vec) => vec
                        .iter()
                        .rev()
                        .map(|&(pi, ti)| Vertex::new(positions[pi], tex_coords[ti]))
                        .collect(),
                    obj::raw::object::Polygon::PTN(ref vec) => vec
                        .iter()
                        .rev()
                        .map(|&(pi, ti, _)| Vertex::new(positions[pi], tex_coords[ti]))
                        .collect(),
                    _ => return Err(anyhow!("Model must only contain polygons with texture UVs")),
                };
                polygons.push(Polygon::new(vertices, texture_ix));
            }
        }
    }

    Ok((polygons, texture_names))
}

fn build_tree(polygons: Vec<Polygon>, mut planes: Vec<ClassifiedPolygons>) -> Vec<Node> {
    // TODO: Heuristically choose a plane to split.
    let plane = match planes.pop() {
        Some(plane) => plane,
        None => return Vec::new(),
    };

    let coplanar = plane
        .coplanar_ixs
        .iter()
        .map(|&i| polygons[i].clone())
        .collect();

    let mut front_planes = Vec::new();
    let mut front_polygons = polygons;
    let mut back_planes = Vec::new();
    let mut back_polygons = front_polygons.clone();

    // Split all the intersected polygons.
    plane
        .intersecting_ixs
        .par_iter()
        .map(|(&index, sides)| {
            let (front_polygon, back_polygon) = front_polygons[index].split(&plane.plane, sides);
            (index, front_polygon, back_polygon)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|(index, front_polygon, back_polygon)| {
            front_polygons[index] = front_polygon;
            back_polygons[index] = back_polygon;
        });

    let reclassify = |mut other_plane: ClassifiedPolygons, polygons: &[Polygon]| {
        plane.intersecting_ixs.iter().for_each(|(&index, _)| {
            other_plane.reclassify_split(index, &polygons[index]);
        });
        other_plane
    };

    for other_plane in planes.into_iter() {
        match other_plane.intersection(&plane.front_ixs) {
            Some(intersection) => front_planes.push(reclassify(intersection, &front_polygons)),
            None => (),
        }
        match other_plane.intersection(&plane.back_ixs) {
            Some(intersection) => back_planes.push(reclassify(intersection, &back_polygons)),
            None => (),
        }
    }

    let (front_children, back_children) = rayon::join(
        || build_tree(front_polygons, front_planes),
        || build_tree(back_polygons, back_planes),
    );

    let children = [
        if front_children.is_empty() {
            None
        } else {
            Some(0)
        },
        if back_children.is_empty() {
            None
        } else {
            Some(front_children.len())
        },
    ];

    let head = Node::new(coplanar, plane.plane.normal, plane.plane.distance, children);
    let mut tree = vec![head];
    tree.extend(front_children);
    tree.extend(back_children);
    tree
}
