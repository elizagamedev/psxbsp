use crate::geometry::Plane;
use float_cmp::approx_eq;
use itertools::izip;
use nalgebra::Vector3;
use std::borrow::Borrow;
use std::collections::HashSet;

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

fn dedup_vector3s<I, R>(it: I) -> Vec<HashSet<usize>>
where
    I: Iterator<Item = R> + Clone,
    R: Borrow<Vector3<f64>>,
{
    let x_equals = dedup_f64s(it.clone().map(|v| v.borrow().x));
    let y_equals = dedup_f64s(it.clone().map(|v| v.borrow().y));
    let z_equals = dedup_f64s(it.map(|v| v.borrow().z));

    izip!(x_equals, y_equals, z_equals)
        .map(|(x, y, z)| {
            x.into_iter()
                .filter(|k| y.contains(k))
                .filter(|k| z.contains(k))
                .collect()
        })
        .collect()
}

pub fn dedup_planes<I, R>(it: I) -> Vec<HashSet<usize>>
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
