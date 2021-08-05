//! This module converts a Wavefront OBJ to a basic auto-partitioned BSP tree.
//! Each node in the tree describes a plane and list of simple polygons, each
//! with an arbitrary number of vertices, which are guaranteed to be coplanar.

use crate::geometry::{OrthonormalBasis2D, Plane, Vertex};
use anyhow::{anyhow, Result};
use float_cmp::approx_eq;
use itertools::izip;
use nalgebra::{Vector2, Vector3};
use obj::raw::RawObj;
use rayon::prelude::*;
use std::borrow::Borrow;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{HashMap, HashSet};

pub struct BasicBspTree {
    pub nodes: Vec<BasicBspNode>,
    pub texture_names: Vec<String>,
}

impl BasicBspTree {
    pub fn from_wavefront_obj(obj: RawObj) -> Result<Self> {
        let (polygons, texture_names) = collect_polygons_from_obj(obj)?;

        let planes: Vec<Plane> = polygons
            .par_iter()
            .map(|polygon| polygon.to_plane())
            .collect();

        if izip!(&polygons, &planes).any(|(polygon, plane)| !polygon.is_planar(plane)) {
            return Err(anyhow!("Non-planar polygon encountered"));
        }

        let collapsed_planes = collapse_planes(&planes);

        let classifications = collapsed_planes
            .into_par_iter()
            .map(|(plane, coplanar_ixs)| {
                PolygonPlaneClassification::new(plane, coplanar_ixs, &polygons)
            })
            .collect();

        let nodes = build_tree(polygons, classifications);

        Ok(BasicBspTree {
            nodes,
            texture_names,
        })
    }
}

#[derive(Clone, Copy)]
enum PlaneSide {
    Front,
    Back,
    Coplanar,
}

#[derive(Clone)]
pub struct BasicBspPolygon {
    pub vertices: Vec<Vertex>,
    pub texture_ix: usize,
}

impl BasicBspPolygon {
    fn new(vertices: Vec<Vertex>, texture_ix: usize) -> Self {
        Self {
            vertices,
            texture_ix,
        }
    }

    fn to_plane(&self) -> Plane {
        // Use Newell's method to determine a plane equation so that concave
        // polygons are correctly computed.
        let mut normal = Vector3::zeros();
        let mut point_a = &self.vertices.last().unwrap().position;
        for point_b in self.vertices.iter().map(|x| &x.position) {
            normal.x += (point_a.z + point_b.z) * (point_b.y - point_a.y);
            normal.y += (point_a.x + point_b.x) * (point_b.z - point_a.z);
            normal.z += (point_a.y + point_b.y) * (point_b.x - point_a.x);
            point_a = point_b;
        }
        normal.normalize_mut();

        let distance = normal.dot(&self.vertices[0].position);
        Plane::new(normal, distance)
    }

    /// Return true if the polygon is perfectly planar. This function assumes
    /// the given plane is coplanar with the first three vertices of the
    /// polygon.
    fn is_planar(&self, plane: &Plane) -> bool {
        if self.vertices.len() == 3 {
            return true;
        }
        self.vertices.iter().skip(3).all(|x| {
            approx_eq!(
                f64,
                plane.normal.dot(&x.position),
                plane.distance,
                epsilon = 0.01,
                ulps = 2i64.pow(48)
            )
        })
    }

    /// Split the polygon along the given plane. Returns a tuple of (front
    /// polygon, back polygon).
    fn split(&self, plane: &Plane, sides: &[PlaneSide]) -> (Self, Self) {
        let mut front_polygon = Vec::new();
        let mut back_polygon = Vec::new();

        let mut side_a = sides.last().unwrap();
        let mut point_a = self.vertices.last().unwrap();

        for (side_b, point_b) in sides.iter().zip(&self.vertices) {
            let mut insert_new_vertex = || {
                let v = point_b.position - point_a.position;
                let sect =
                    -(plane.normal.dot(&point_a.position) - plane.distance) / plane.normal.dot(&v);
                let new_position = point_a.position + v * sect;
                let new_tex_coord =
                    point_a.tex_coord + (point_b.tex_coord - point_a.tex_coord) * sect;
                front_polygon.push(Vertex::new(new_position, new_tex_coord));
                back_polygon.push(Vertex::new(new_position, new_tex_coord));
            };

            match side_b {
                PlaneSide::Front => {
                    if matches!(side_a, PlaneSide::Back) {
                        insert_new_vertex();
                    }
                    front_polygon.push(point_b.clone());
                }
                PlaneSide::Back => {
                    if matches!(side_a, PlaneSide::Front) {
                        insert_new_vertex();
                    }
                    back_polygon.push(point_b.clone());
                }
                PlaneSide::Coplanar => {
                    front_polygon.push(point_b.clone());
                    back_polygon.push(point_b.clone());
                }
            }

            side_a = side_b;
            point_a = point_b;
        }

        (
            BasicBspPolygon::new(front_polygon, self.texture_ix),
            BasicBspPolygon::new(back_polygon, self.texture_ix),
        )
    }
}

struct PolygonPlaneClassification {
    plane: Plane,
    coplanar_ixs: HashSet<usize>,
    front_ixs: HashSet<usize>,
    back_ixs: HashSet<usize>,
    intersecting_ixs: HashMap<usize, Vec<PlaneSide>>,
    basis: OrthonormalBasis2D,
}

impl PolygonPlaneClassification {
    fn new(
        plane: Plane,
        coplanar_ixs: HashSet<usize>,
        polygons: &[BasicBspPolygon],
    ) -> PolygonPlaneClassification {
        let basis = OrthonormalBasis2D::from_plane(&plane);
        let mut result = PolygonPlaneClassification {
            plane,
            coplanar_ixs,
            front_ixs: HashSet::new(),
            back_ixs: HashSet::new(),
            intersecting_ixs: HashMap::new(),
            basis,
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
    fn classify(&mut self, polygon_ix: usize, polygon: &BasicBspPolygon) {
        let sides: Vec<PlaneSide> = polygon
            .vertices
            .iter()
            .map(|v| {
                let this_distance = self.plane.normal.dot(&v.position);
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
    fn reclassify_split(&mut self, polygon_ix: usize, polygon: &BasicBspPolygon) {
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

        Some(PolygonPlaneClassification {
            plane: self.plane.clone(),
            coplanar_ixs,
            front_ixs,
            back_ixs,
            intersecting_ixs,
            basis: self.basis.clone(),
        })
    }
}

pub struct BasicBspNode {
    pub polygons: Vec<BasicBspPolygon>,
    pub plane: Plane,
    pub front_child: Option<usize>,
    pub back_child: Option<usize>,
    // We don't need the basis for the BSP stage, but we calculate it here so
    // that we can do fewer calculations down the line.
    pub basis: OrthonormalBasis2D,
}

impl BasicBspNode {
    fn new(
        polygons: Vec<BasicBspPolygon>,
        plane: Plane,
        front_child: Option<usize>,
        back_child: Option<usize>,
        basis: OrthonormalBasis2D,
    ) -> Self {
        Self {
            polygons,
            plane,
            front_child,
            back_child,
            basis,
        }
    }
}

fn collect_polygons_from_obj(obj: obj::raw::RawObj) -> Result<(Vec<BasicBspPolygon>, Vec<String>)> {
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
                polygons.push(BasicBspPolygon::new(vertices, texture_ix));
            }
        }
    }

    Ok((polygons, texture_names))
}

fn build_tree(
    polygons: Vec<BasicBspPolygon>,
    mut planes: Vec<PolygonPlaneClassification>,
) -> Vec<BasicBspNode> {
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

    let reclassify = |mut other_plane: PolygonPlaneClassification, polygons: &[BasicBspPolygon]| {
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

    let front_child = if front_children.is_empty() {
        None
    } else {
        Some(0)
    };
    let back_child = if back_children.is_empty() {
        None
    } else {
        Some(front_children.len())
    };

    let head = BasicBspNode::new(coplanar, plane.plane, front_child, back_child, plane.basis);
    let mut tree = vec![head];
    tree.extend(front_children);
    tree.extend(back_children);
    tree
}

/// Given a list of planes, combine all coplanar entries. The return value is a
/// list of tuples associating the average Plane value to the set of indices of
/// all coplanar entries of the original list.
fn collapse_planes(planes: &[Plane]) -> Vec<(Plane, HashSet<usize>)> {
    // Create a list of values.
    let mut value_ixs = vec![0; planes.len()];
    let mut values = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    for (i, duplicates) in dedup_planes(planes.iter()).into_iter().enumerate() {
        if visited.len() == planes.len() {
            // No need to continue iterating if we've visited every vector.
            break;
        }
        if visited.contains(&i) {
            continue;
        }
        visited.extend(&duplicates);

        // Store the index for every duplicate.
        value_ixs[i] = values.len();
        for j in duplicates.iter() {
            value_ixs[*j] = values.len();
        }

        // Calculate average of all duplicates to be the canonical value.
        let len = (duplicates.len() + 1) as f64;
        let normal = (planes[i].normal
            + duplicates
                .iter()
                .map(|&x| planes[x].normal)
                .sum::<Vector3<f64>>())
            / len;
        let distance = (planes[i].distance
            + duplicates.iter().map(|&x| planes[x].distance).sum::<f64>())
            / len;
        values.push(Plane::new(normal, distance));
    }

    // Generate the final result.
    let mut map = HashMap::new();
    for (ix, value_ix) in value_ixs.iter().enumerate() {
        map.entry(value_ix).or_insert(HashSet::new()).insert(ix);
    }
    map.into_iter()
        .map(|(&ix, set)| (values[ix].clone(), set))
        .collect()
}

fn dedup_f64s<I, R>(it: I) -> Vec<HashSet<usize>>
where
    I: Iterator<Item = R> + Clone,
    R: Borrow<f64>,
{
    let mut sorted_values: Vec<(usize, f64)> =
        it.into_iter().map(|x| *x.borrow()).enumerate().collect();
    sorted_values.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    let mut result = vec![HashSet::new(); sorted_values.len()];
    let mut lhs = 0;
    let mut rhs = 1;
    while lhs < sorted_values.len() && rhs < sorted_values.len() {
        let (lhs_index, lhs_value) = sorted_values[lhs];
        let (rhs_index, rhs_value) = sorted_values[rhs];
        if approx_eq!(
            f64,
            lhs_value,
            rhs_value,
            epsilon = 0.01,
            ulps = 2i64.pow(48)
        ) {
            result[lhs_index].insert(rhs_index);
            result[rhs_index].insert(lhs_index);
            rhs += 1;
        } else {
            lhs += 1;
            rhs = lhs + 1;
        }
    }
    result
}

fn dedup_planes<I, R>(it: I) -> Vec<HashSet<usize>>
where
    I: Iterator<Item = R> + Clone,
    R: Borrow<Plane>,
{
    let x_equals = dedup_f64s(it.clone().map(|v| v.borrow().normal.x));
    let y_equals = dedup_f64s(it.clone().map(|v| v.borrow().normal.y));
    let z_equals = dedup_f64s(it.clone().map(|v| v.borrow().normal.z));
    let d_equals = dedup_f64s(it.map(|v| v.borrow().distance));

    izip!(x_equals, y_equals, z_equals, d_equals)
        .map(|(x, y, z, d)| {
            x.into_iter()
                .filter(|k| y.contains(k))
                .filter(|k| z.contains(k))
                .filter(|k| d.contains(k))
                .collect()
        })
        .collect()
}
