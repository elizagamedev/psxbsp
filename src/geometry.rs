use std::collections::{HashMap, HashSet};

use float_cmp::approx_eq;
use nalgebra::{Vector2, Vector3};
use serde::Serialize;

use crate::dedup::dedup_planes;

#[derive(Serialize, Debug, Clone, Copy)]
pub enum PlaneSide {
    Front,
    Back,
    Coplanar,
}

#[derive(Serialize, Debug, Clone)]
pub struct Vertex {
    pub position: Vector3<f64>,
    pub tex_coord: Vector2<f64>,
}

impl Vertex {
    pub fn new(position: Vector3<f64>, tex_coord: Vector2<f64>) -> Self {
        Self {
            position,
            tex_coord,
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct Polygon {
    pub vertices: Vec<Vertex>,
    pub texture_ix: usize,
}

impl Polygon {
    pub fn new(vertices: Vec<Vertex>, texture_ix: usize) -> Self {
        Self {
            vertices,
            texture_ix,
        }
    }

    pub fn to_plane(&self) -> Plane {
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

        // let p1 = self.vertices[0].position;
        // let p2 = self.vertices[1].position;
        // let p3 = self.vertices[2].position;
        // let normal = (p2 - p1).cross(&(p3 - p1)).normalize();

        let distance = normal.dot(&self.vertices[0].position);
        Plane::new(normal, distance)
    }

    /// Return true if the polygon is perfectly planar. This function assumes
    /// the given plane is coplanar with the first three vertices of the
    /// polygon.
    pub fn is_planar(&self, plane: &Plane) -> bool {
        if self.vertices.len() == 3 {
            return true;
        }
        self.vertices.iter().skip(3).all(|x| {
            approx_eq!(
                f64,
                x.position.dot(&plane.normal),
                plane.distance,
                epsilon = 0.01,
                ulps = 2i64.pow(48)
            )
        })
    }

    /// Split the polygon along the given plane. Returns a tuple of (front
    /// polygon, back polygon).
    pub fn split(&self, plane: &Plane, sides: &[PlaneSide]) -> (Self, Self) {
        let mut front_polygon = Vec::new();
        let mut back_polygon = Vec::new();

        let mut side_a = sides.last().unwrap();
        let mut point_a = self.vertices.last().unwrap();

        for (side_b, point_b) in sides.iter().zip(&self.vertices) {
            let mut insert_new_vertex = || {
                let v = point_b.position - point_a.position;
                let sect =
                    -(point_a.position.dot(&plane.normal) - plane.distance) / plane.normal.dot(&v);
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
            Polygon::new(front_polygon, self.texture_ix),
            Polygon::new(back_polygon, self.texture_ix),
        )
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct Plane {
    pub normal: Vector3<f64>,
    pub distance: f64,
}

impl Plane {
    pub fn new(normal: Vector3<f64>, distance: f64) -> Self {
        Self { normal, distance }
    }
}

/// Given a list of planes, combine all coplanar entries. The return value is a
/// list of tuples associating the average Plane value to the set of indices of
/// all coplanar entries of the original list.
pub fn collapse_planes(planes: &[Plane]) -> Vec<(Plane, HashSet<usize>)> {
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
