use anyhow::{anyhow, Result};
use float_cmp::approx_eq;
use itertools::izip;
use nalgebra::{vector, Vector2, Vector3};
use obj::raw::parse_obj;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

struct Vertex {
    position: usize,
    tex_coord: usize,
}

impl Vertex {
    fn new(position: usize, tex_coord: usize) -> Self {
        Self {
            position,
            tex_coord,
        }
    }
}

struct Polygon {
    vertices: Vec<Vertex>,
    normal: usize,
}

impl Polygon {
    fn new(vertices: Vec<Vertex>, normal: usize) -> Self {
        Self { vertices, normal }
    }
}

struct Mesh {
    positions: Vec<Vector3<f32>>,
    tex_coords: Vec<Vector2<f32>>,
    normals: Vec<Vector3<f32>>,
    polygons: Vec<Polygon>,
}

impl Mesh {
    fn new(
        positions: Vec<Vector3<f32>>,
        tex_coords: Vec<Vector2<f32>>,
        normals: Vec<Vector3<f32>>,
        polygons: Vec<Polygon>,
    ) -> Self {
        Self {
            positions,
            tex_coords,
            normals,
            polygons,
        }
    }
}

/// Given a list of floats, construct a map of float indices to sets of
/// approximately equal float indices.
fn dedup_floats<I>(values: I) -> Vec<HashSet<usize>>
where
    I: Iterator<Item = f32>,
{
    let mut sorted_values: Vec<(usize, f32)> = values.enumerate().collect();
    sorted_values.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    let mut result = vec![HashSet::new(); sorted_values.len()];
    let mut lhs = 0;
    let mut rhs = 1;
    while lhs < sorted_values.len() && rhs < sorted_values.len() {
        let (lhs_index, lhs_value) = sorted_values[lhs];
        let (rhs_index, rhs_value) = sorted_values[rhs];
        if approx_eq!(f32, lhs_value, rhs_value, ulps = 4) {
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

/// Given a list of vectors, construct a map of vector indices to sets of
/// approximately equal vector indices.
fn dedup_vectors(vectors: &Vec<Vector3<f32>>) -> Vec<HashSet<usize>> {
    let x_equals = dedup_floats(vectors.iter().map(|v| v.x));
    let y_equals = dedup_floats(vectors.iter().map(|v| v.y));
    let z_equals = dedup_floats(vectors.iter().map(|v| v.z));

    izip!(x_equals, y_equals, z_equals)
        .map(|(x, y, z)| {
            x.into_iter()
                .filter(|k| y.contains(k))
                .filter(|k| z.contains(k))
                .collect()
        })
        .collect()
}

/// Given a list of vectors, removes duplicates and returns a tuple of (list of
/// vector indices, list of vector values).
fn collapse_vectors(vectors: Vec<Vector3<f32>>) -> (Vec<usize>, Vec<Vector3<f32>>) {
    let mut indices = vec![0; vectors.len()];
    let mut values = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    for (i, duplicates) in dedup_vectors(&vectors).into_iter().enumerate() {
        if visited.len() == vectors.len() {
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
        let len = (duplicates.len() + 1) as f32;
        let value = (duplicates
            .into_iter()
            .map(|x| vectors[x])
            .sum::<Vector3<f32>>()
            + vectors[i])
            / len;
        values.push(value);
    }
    (indices, values)
}

/// Given a list of polygons, returns a list of normal values.
fn calculate_normals(
    positions: &Vec<Vector3<f32>>,
    polygons: &Vec<Vec<Vertex>>,
) -> Vec<Vector3<f32>> {
    polygons
        .iter()
        .map(|vertices| {
            let p1 = positions[vertices[0].position];
            let p2 = positions[vertices[1].position];
            let p3 = positions[vertices[2].position];
            (p2 - p1).cross(&(p3 - p1)).normalize()
        })
        .collect()
}

fn load_mesh<T: BufRead>(input: T) -> Result<Mesh> {
    let obj = parse_obj(input)?;

    let positions: Vec<Vector3<f32>> = obj
        .positions
        .iter()
        .map(|x| vector![x.0, x.1, x.2])
        .collect();

    let tex_coords = obj.tex_coords.iter().map(|x| vector![x.0, x.1]).collect();

    let mut raw_polygons: Vec<Vec<Vertex>> = Vec::with_capacity(obj.polygons.len());

    for polygon in obj.polygons {
        match polygon {
            obj::raw::object::Polygon::PT(ref vec) => {
                raw_polygons.push(vec.iter().map(|&(pi, ti)| Vertex::new(pi, ti)).collect());
            }
            obj::raw::object::Polygon::PTN(ref vec) => {
                raw_polygons.push(vec.iter().map(|&(pi, ti, _)| Vertex::new(pi, ti)).collect());
            }
            _ => return Err(anyhow!("Model must only contain polygons with texture UVs")),
        }
    }

    let raw_normal_values = calculate_normals(&positions, &raw_polygons);
    let (normal_indices, normals) = collapse_vectors(raw_normal_values);

    let polygons = raw_polygons
        .into_iter()
        .zip(normal_indices)
        .map(|(vertices, normals)| Polygon::new(vertices, normals))
        .collect();

    Ok(Mesh::new(positions, tex_coords, normals, polygons))
}

fn work() -> Result<()> {
    let reader = BufReader::new(File::open("/home/eliza/bob.obj")?);
    let mesh = load_mesh(reader)?;
    Ok(())
}

fn main() {
    work().unwrap();
}
