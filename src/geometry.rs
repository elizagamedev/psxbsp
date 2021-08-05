use float_cmp::approx_eq;
use nalgebra::{Vector2, Vector3};
use serde::Serialize;

#[derive(Clone, Serialize)]
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

#[derive(Clone, Serialize)]
pub struct Plane {
    pub normal: Vector3<f64>,
    pub distance: f64,
}

impl Plane {
    pub fn new(normal: Vector3<f64>, distance: f64) -> Self {
        Self { normal, distance }
    }
}

#[derive(Clone)]
pub struct OrthonormalBasis2D {
    x: Vector3<f64>,
    y: Vector3<f64>,
}

impl OrthonormalBasis2D {
    pub fn from_plane(plane: &Plane) -> Self {
        // Create orthonormal basis with respect to the plane.
        let mut u = Vector3::z();
        let mut uoz = u.dot(&plane.normal);
        if approx_eq!(f64, uoz, 1.0, epsilon = 0.01, ulps = 2i64.pow(48)) {
            u = Vector3::y();
            uoz = u.dot(&plane.normal);
        }
        let x = (u - plane.normal * uoz).normalize();
        let y = plane.normal.cross(&x);
        Self { x, y }
    }

    pub fn transform(&self, vector3: &Vector3<f64>) -> Vec<f64> {
        vec![self.x.dot(&vector3), self.y.dot(&vector3)]
    }
}
