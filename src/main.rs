use anyhow::{anyhow, Result};
use float_cmp::approx_eq;
use itertools::izip;
use nalgebra::{vector, Vector2, Vector3};
use obj::raw::{parse_obj, RawObj};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::iter::Sum;
use std::ops::{Add, Div};

type Float = f64;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Vertex {
    position: Vector3<Float>,
    tex_coord: Vector2<Float>,
}

impl Vertex {
    fn new(position: Vector3<Float>, tex_coord: Vector2<Float>) -> Self {
        Self {
            position,
            tex_coord,
        }
    }
}

#[derive(Clone, Copy)]
enum PlaneSide {
    Front,
    Back,
}

struct Plane {
    coplanar: HashSet<usize>,
    front: HashSet<usize>,
    back: HashSet<usize>,
    intersecting: HashMap<usize, Vec<PlaneSide>>,
    normal: Vector3<Float>,
    distance: Float,
}

impl Plane {
    fn new(
        coplanar: HashSet<usize>,
        front: HashSet<usize>,
        back: HashSet<usize>,
        intersecting: HashMap<usize, Vec<PlaneSide>>,
        normal: Vector3<Float>,
        distance: Float,
    ) -> Self {
        Self {
            coplanar,
            front,
            back,
            intersecting,
            normal,
            distance,
        }
    }

    fn into_intersection(self, other: &HashSet<usize>) -> Self {
        let into_intersection =
            |set: HashSet<usize>| set.into_iter().filter(|x| other.contains(x)).collect();
        let coplanar = into_intersection(self.coplanar);
        let front = into_intersection(self.front);
        let back = into_intersection(self.back);
        let intersecting = self
            .intersecting
            .into_iter()
            .filter(|(k, _)| other.contains(k))
            .collect();
        Plane::new(
            coplanar,
            front,
            back,
            intersecting,
            self.normal,
            self.distance,
        )
    }

    fn intersection(&self, other: &HashSet<usize>) -> Self {
        let coplanar = self.coplanar.intersection(other).copied().collect();
        let front = self.front.intersection(other).copied().collect();
        let back = self.back.intersection(other).copied().collect();
        let intersecting = self
            .intersecting
            .iter()
            .filter(|(k, _)| other.contains(k))
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        Plane::new(
            coplanar,
            front,
            back,
            intersecting,
            self.normal,
            self.distance,
        )
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Node {
    polygons: Vec<Vec<Vertex>>,
    normal: Vector3<Float>,
    distance: Float,
    children: [Option<usize>; 2],
}

impl Node {
    fn new(
        polygons: Vec<Vec<Vertex>>,
        normal: Vector3<Float>,
        distance: Float,
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

enum DedupType {}

trait Dedup<T> {
    /// Given a list of values, construct a map of indices to sets of
    /// approximately equal indices.
    fn dedup<I, R>(it: I) -> Vec<HashSet<usize>>
    where
        I: Iterator<Item = R> + Clone,
        R: Borrow<T>;
}

impl Dedup<Float> for DedupType {
    fn dedup<I, R>(it: I) -> Vec<HashSet<usize>>
    where
        I: Iterator<Item = R> + Clone,
        R: Borrow<Float>,
    {
        let mut sorted_values: Vec<(usize, Float)> =
            it.into_iter().map(|x| *x.borrow()).enumerate().collect();
        sorted_values.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        let mut result = vec![HashSet::new(); sorted_values.len()];
        let mut lhs = 0;
        let mut rhs = 1;
        while lhs < sorted_values.len() && rhs < sorted_values.len() {
            let (lhs_index, lhs_value) = sorted_values[lhs];
            let (rhs_index, rhs_value) = sorted_values[rhs];
            if approx_eq!(
                Float,
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
}

impl Dedup<Vector3<Float>> for DedupType {
    fn dedup<I, R>(it: I) -> Vec<HashSet<usize>>
    where
        I: Iterator<Item = R> + Clone,
        R: Borrow<Vector3<Float>>,
    {
        let x_equals = DedupType::dedup(it.clone().map(|v| v.borrow().x));
        let y_equals = DedupType::dedup(it.clone().map(|v| v.borrow().y));
        let z_equals = DedupType::dedup(it.map(|v| v.borrow().z));

        izip!(x_equals, y_equals, z_equals)
            .map(|(x, y, z)| {
                x.into_iter()
                    .filter(|k| y.contains(k))
                    .filter(|k| z.contains(k))
                    .collect()
            })
            .collect()
    }
}

/// Given a list of vectors, removes duplicates and returns a tuple of (list of
/// vector indices, list of vector values).
fn collapse<T, D>(entries: &[T]) -> (Vec<usize>, Vec<T>)
where
    T: Add<T, Output = T> + Div<Float, Output = T> + Sum + Copy,
    D: Dedup<T>,
{
    let mut indices = vec![0; entries.len()];
    let mut values = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    for (i, duplicates) in D::dedup(entries.iter()).into_iter().enumerate() {
        if visited.len() == entries.len() {
            // No need to continue iterating if we've visited every vector.
            break;
        }
        if visited.contains(&i) {
            continue;
        }
        visited.extend(&duplicates);

        // Store the index for every duplicate.
        indices[i] = values.len();
        for j in duplicates.iter() {
            indices[*j] = values.len();
        }

        // Calculate average of all duplicates to be the canonical value.
        let len = (duplicates.len() + 1) as Float;
        let value = (entries[i] + duplicates.into_iter().map(|x| entries[x]).sum()) / len;
        values.push(value);
    }
    (indices, values)
}

/// Given a list of polygons, returns a list of normal and d values.
fn calculate_raw_normals_and_distances(
    polygons: &[Vec<Vertex>],
) -> (Vec<Vector3<Float>>, Vec<Float>) {
    let normals: Vec<_> = polygons
        .par_iter()
        .map(|vertices| {
            let p1 = vertices[0].position;
            let p2 = vertices[1].position;
            let p3 = vertices[2].position;
            (p2 - p1).cross(&(p3 - p1)).normalize()
        })
        .collect();
    let distances = polygons
        .par_iter()
        .zip(&normals)
        .map(|(vertices, normal)| {
            let p1 = vertices[0].position;
            normal.dot(&p1)
        })
        .collect();
    (normals, distances)
}

fn is_polygon_planar(polygon: &Vec<Vertex>, normal: &Vector3<Float>, d: Float) -> bool {
    if polygon.len() == 3 {
        return true;
    }
    polygon.iter().skip(3).all(|x| {
        approx_eq!(
            Float,
            x.position.dot(&normal),
            d,
            epsilon = 0.01,
            ulps = 2i64.pow(48)
        )
    })
}

fn collect_polygons(obj: obj::raw::RawObj) -> Result<Vec<Vec<Vertex>>> {
    let positions: Vec<_> = obj
        .positions
        .iter()
        .map(|x| vector![x.0 as Float, x.1 as Float, x.2 as Float])
        .collect();

    let tex_coords: Vec<_> = obj
        .tex_coords
        .iter()
        .map(|x| vector![x.0 as Float, x.1 as Float])
        .collect();

    let mut polygons: Vec<Vec<Vertex>> = Vec::with_capacity(obj.polygons.len());

    for polygon in obj.polygons {
        match polygon {
            obj::raw::object::Polygon::PT(ref vec) => {
                polygons.push(
                    vec.iter()
                        .map(|&(pi, ti)| Vertex::new(positions[pi], tex_coords[ti]))
                        .collect(),
                );
            }
            obj::raw::object::Polygon::PTN(ref vec) => {
                polygons.push(
                    vec.iter()
                        .map(|&(pi, ti, _)| Vertex::new(positions[pi], tex_coords[ti]))
                        .collect(),
                );
            }
            _ => return Err(anyhow!("Model must only contain polygons with texture UVs")),
        }
    }

    Ok(polygons)
}

fn calculate_plane_polygon_relationship(
    index: usize,
    polygon: &Vec<Vertex>,
    normal: &Vector3<Float>,
    distance: Float,
    front: &mut HashSet<usize>,
    back: &mut HashSet<usize>,
    intersecting: &mut HashMap<usize, Vec<PlaneSide>>,
) {
    let vertex_sides: Vec<PlaneSide> = polygon
        .iter()
        .map(|v| {
            if v.position.dot(&normal) < distance {
                PlaneSide::Front
            } else {
                PlaneSide::Back
            }
        })
        .collect();
    if vertex_sides
        .iter()
        .all(|side| matches!(*side, PlaneSide::Front))
    {
        front.insert(index);
    } else if vertex_sides
        .iter()
        .all(|side| matches!(*side, PlaneSide::Back))
    {
        back.insert(index);
    } else {
        intersecting.insert(index, vertex_sides);
    }
}

fn calculate_plane(
    polygons: &[Vec<Vertex>],
    normal: Vector3<Float>,
    distance: Float,
    coplanar: HashSet<usize>,
) -> Plane {
    let mut front = HashSet::new();
    let mut back = HashSet::new();
    let mut intersecting = HashMap::new();

    for (i, other) in polygons.iter().enumerate() {
        calculate_plane_polygon_relationship(
            i,
            other,
            &normal,
            distance,
            &mut front,
            &mut back,
            &mut intersecting,
        );
    }

    Plane::new(coplanar, front, back, intersecting, normal, distance)
}

fn collect_coplanar_polygons(
    normal_indices: &[usize],
    distance_indices: &[usize],
) -> Vec<HashSet<usize>> {
    let mut map = HashMap::new();
    for (i, (normal_index, distance_index)) in
        normal_indices.iter().zip(distance_indices).enumerate()
    {
        map.entry((normal_index, distance_index))
            .or_insert(HashSet::new())
            .insert(i);
    }
    map.into_iter().map(|(_, set)| set).collect()
}

struct Bounds {
    mins: Vector3<Float>,
    maxs: Vector3<Float>,
}

impl Bounds {
    fn new(mins: Vector3<Float>, maxs: Vector3<Float>) -> Self {
        Self { mins, maxs }
    }
}

fn calculate_bounds(positions: &[Vector3<Float>]) -> Bounds {
    let mut mins = vector![Float::INFINITY, Float::INFINITY, Float::INFINITY];
    let mut maxs = vector![
        Float::NEG_INFINITY,
        Float::NEG_INFINITY,
        Float::NEG_INFINITY
    ];

    for position in positions {
        if position.x > maxs.x {
            maxs.x = position.x;
        }
        if position.x < mins.x {
            mins.x = position.x;
        }
        if position.y > maxs.y {
            maxs.y = position.y;
        }
        if position.y < mins.y {
            mins.y = position.y;
        }
        if position.z > maxs.z {
            maxs.z = position.z;
        }
        if position.z < mins.z {
            mins.z = position.z;
        }
    }

    assert_ne!(mins.x, Float::INFINITY);
    assert_ne!(mins.y, Float::INFINITY);
    assert_ne!(mins.z, Float::INFINITY);
    assert_ne!(maxs.x, Float::NEG_INFINITY);
    assert_ne!(maxs.y, Float::NEG_INFINITY);
    assert_ne!(maxs.z, Float::NEG_INFINITY);

    return Bounds::new(mins, maxs);
}

fn split_polygon(
    normal: &Vector3<Float>,
    distance: Float,
    sides: &[PlaneSide],
    polygon: &[Vertex],
) -> (Vec<Vertex>, Vec<Vertex>) {
    let mut front_polygon = Vec::new();
    let mut back_polygon = Vec::new();

    let mut side_a = sides.last().unwrap();
    let mut point_a = polygon.last().unwrap();

    for (side_b, point_b) in sides.iter().zip(polygon) {
        let mut insert_new_vertex = || {
            let v = point_b.position - point_a.position;
            let sect = -(point_a.position.dot(normal) + distance) / normal.dot(&v);
            let new_point = point_a.position + (v * sect);
            front_polygon.push(Vertex::new(new_point, vector![0.0, 0.0]));
            back_polygon.push(Vertex::new(new_point, vector![0.0, 0.0]));
        };

        match side_b {
            PlaneSide::Front => {
                if matches!(side_a, PlaneSide::Back) {
                    insert_new_vertex();
                }
                front_polygon.push(point_b.clone());
            }
            PlaneSide::Back => {
                if matches!(side_b, PlaneSide::Front) {
                    insert_new_vertex();
                }
                back_polygon.push(point_b.clone());
            }
        }

        side_a = side_b;
        point_a = point_b;
    }

    (front_polygon, back_polygon)
}

fn recalculate_intersection(
    front_polygons: &[Vec<Vertex>],
    back_polygons: &[Vec<Vertex>],
    polygon_index: usize,
    plane: &mut Plane,
) {
    match plane.intersecting.entry(polygon_index) {
        Occupied(entry) => entry.remove_entry(),
        Vacant(_) => return,
    };

    let front_polygon = &front_polygons[polygon_index];
    let back_polygon = &back_polygons[polygon_index];

    // TODO: We only really need to recalculate two of these values.

    let mut recalculate = |polygon: &Vec<Vertex>| {
        calculate_plane_polygon_relationship(
            polygon_index,
            &polygon,
            &plane.normal,
            plane.distance,
            &mut plane.front,
            &mut plane.back,
            &mut plane.intersecting,
        );
    };

    recalculate(front_polygon);
    recalculate(back_polygon);
}

fn build_tree(polygons: Vec<Vec<Vertex>>, mut planes: Vec<Plane>) -> Vec<Node> {
    // TODO: Heuristically choose a plane to split.
    let plane = match planes.pop() {
        Some(plane) => plane,
        None => return Vec::new(),
    };

    let normal = plane.normal;
    let distance = plane.distance;

    let coplanar = plane
        .coplanar
        .into_iter()
        .map(|i| polygons[i].clone())
        .collect();

    let mut front_planes = Vec::new();
    let mut front_polygons = polygons;
    let mut back_planes = Vec::new();
    let mut back_polygons = front_polygons.clone();

    // Split all the intersected polygons.
    plane
        .intersecting
        .par_iter()
        .map(|(&index, sides)| {
            let (front_polygon, back_polygon) =
                split_polygon(&normal, distance, &sides, &front_polygons[index]);
            (index, front_polygon, back_polygon)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|(index, front_polygon, back_polygon)| {
            front_polygons[index] = front_polygon;
            back_polygons[index] = back_polygon;
        });

    for mut other_plane in planes.into_iter() {
        // For all of the polygons split above, recalculate their PlaneSides with respect to every other plane.
        plane.intersecting.iter().for_each(|(&index, _)| {
            recalculate_intersection(&front_polygons, &back_polygons, index, &mut other_plane);
        });
        if plane.front.is_superset(&other_plane.coplanar) {
            front_planes.push(other_plane.into_intersection(&plane.front));
        } else if plane.back.is_superset(&other_plane.coplanar) {
            back_planes.push(other_plane.into_intersection(&plane.back));
        } else {
            front_planes.push(other_plane.intersection(&plane.front));
            back_planes.push(other_plane.into_intersection(&plane.back));
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

    let head = Node::new(coplanar, plane.normal, plane.distance, children);
    let mut tree = vec![head];
    tree.extend(front_children);
    tree.extend(back_children);
    tree
}

fn load_mesh(obj: RawObj) -> Result<()> {
    let polygons = collect_polygons(obj)?;

    let (raw_normal_values, raw_distance_values) = calculate_raw_normals_and_distances(&polygons);
    if izip!(&polygons, &raw_normal_values, &raw_distance_values)
        .map(|(p, n, d)| (p, n, *d))
        .collect::<Vec<(&Vec<Vertex>, &Vector3<Float>, Float)>>()
        .into_iter()
        .any(|(p, n, d)| !is_polygon_planar(p, n, d))
    {
        return Err(anyhow!("Non-planar polygon encountered"));
    }
    let (normal_indices, normals) = collapse::<_, DedupType>(&raw_normal_values);
    let (distance_indices, distances) = collapse::<_, DedupType>(&raw_distance_values);

    let coplanar_polygons = collect_coplanar_polygons(&normal_indices, &distance_indices);
    let planes = izip!(&normal_indices, &distance_indices, coplanar_polygons)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(&normal_index, &distance_index, coplanar)| {
            calculate_plane(
                &polygons,
                normals[normal_index],
                distances[distance_index],
                coplanar,
            )
        })
        .collect();

    let tree = build_tree(polygons, planes);
    println!("{}", serde_yaml::to_string(&tree).unwrap());

    Ok(())
}

fn work() -> Result<()> {
    let obj = parse_obj(BufReader::new(File::open("/home/eliza/bob.obj")?))?;
    let mesh = load_mesh(obj)?;

    Ok(())
}

fn main() {
    work().unwrap();
}
