//! Facing planes: whether two planar faces look at each other across material, how far apart
//! they are, and the plane halfway between them. The geometry under a mid-surface of a thin
//! wall (the face pairing of a wall's two sides).
//!
//! A face's plane is given as a point on it and its outward unit normal, as the face's side of
//! its shell has it. Two planes face each other across material when their outward normals run
//! opposite ways (within the kernel's angular tolerance, [`Tolerances::ang_tol_rad`]) and each
//! lies behind the other: the material is between them, so the outward normals point away from
//! each other. Planes whose outward normals point toward each other bound air, not a wall, and
//! two planes at the same place bound nothing between them: neither pairs.
//!
//! The thickness is measured along the first plane's normal from its point to where that line
//! meets the second plane (so any point of the second plane gives it), and the plane halfway
//! is parallel to it. That is exact when the planes are parallel and stands within the angular
//! tolerance's error otherwise; a pair tilted past the tolerance (a drafted wall) is not paired
//! here: [`drafted_planes`] gives its bisector.
//!
//! Drafted planes: two planes whose outward normals point apart but are tilted past the angular
//! tolerance (a moulded or cast wall, its sides drafted). The plane halfway between them is their
//! bisector: the points as far behind one as behind the other, through the line the two planes
//! meet in. With `da` and `db` how far a point stands in front of each plane (negative behind
//! it), the bisector is `da = db`, the plane with normal `ua - ub` (the two unit outward normals),
//! and the wall's thickness at a point is the length of the bisector's normal line through it
//! between the two planes, `-(da + db) / cos(draft / 2)`: positive where the point lies behind
//! both (in the wall), zero on the line they meet in. Exact: every quantity is a closed form of
//! the two planes. Whether the material is between them where the faces overlap is the caller's
//! to read from the thickness at the overlap's points.
//!
//! Not upstream: no counterpart in SMLib. Deviations: none. No recursion.
//!
//! [`Tolerances::ang_tol_rad`]: crate::kernel::Tolerances::ang_tol_rad

use crate::error::{Result, SolidError};
use crate::geom::vector3d::{Point3, Vec3};
use crate::kernel::Kernel;
use crate::math::det;

const OP: &str = "surface::face_pair::facing_planes";

/// A planar face's plane: a point on it and the face's outward normal (any nonzero length).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutwardPlane {
    /// A point on the plane.
    pub point: Point3,
    /// The face's outward normal, as its shell has the face's side.
    pub outward: Vec3,
}

/// Two planes that face each other across material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FacingPlanes {
    /// The distance between the planes, along the first plane's normal: the wall's thickness.
    /// Always greater than the kernel's zone tolerance.
    pub thickness: f64,
    /// A point on the plane halfway between them.
    pub mid_point: Point3,
    /// The unit normal of the first plane: the halfway plane's side toward the first plane.
    pub toward_first: Vec3,
}

fn finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// Whether `a` and `b` face each other across material, and if so across how much.
///
/// `Ok(None)` is an answer, not a failure: the planes are not opposite within the angular
/// tolerance, or bound air, or are no farther apart than the zone tolerance.
///
/// # Errors
/// [`SolidError::InvalidInput`] for a point or normal that is not finite, or a normal of zero
/// length.
pub fn facing_planes(
    kernel: &Kernel,
    a: &OutwardPlane,
    b: &OutwardPlane,
) -> Result<Option<FacingPlanes>> {
    if !(finite(a.point) && finite(b.point) && finite(a.outward) && finite(b.outward)) {
        return Err(SolidError::InvalidInput {
            operation: OP,
            reason: "a plane's point and normal must be finite",
        });
    }
    let (ua, ub) = (a.outward.unitize()?, b.outward.unitize()?);
    // Opposite normals: pointing apart, and the sine of the angle between them within tolerance.
    let tol = kernel.tolerances;
    if ua.dot(ub) >= 0.0 || ua.cross(ub).length() > det::sin(tol.ang_tol_rad()) {
        return Ok(None);
    }
    // How far behind `a.point`, along `ua`, the normal line through it meets `b`'s plane:
    // `(a.point - t ua - b.point) . ub = 0` gives `t = (a.point - b.point) . ub / (ua . ub)`,
    // whichever point represents `b`. Positive when `b` lies behind `a`: the material is between
    // them. `ua . ub` is below `-cos(ang_tol)` here, so the division is safe.
    let thickness = a.point.distance_along(b.point, ub) / ua.dot(ub);
    if thickness <= tol.zone_tol_3d {
        return Ok(None);
    }
    Ok(Some(FacingPlanes {
        thickness,
        mid_point: a.point - (ua * (thickness * 0.5)),
        toward_first: ua,
    }))
}

const DRAFT_OP: &str = "surface::face_pair::drafted_planes";

/// Two planes whose outward normals point apart, tilted past the angular tolerance: a drafted
/// wall's two sides, and the plane halfway between them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftedPlanes {
    /// The angle between the two planes (radians), more than the angular tolerance and less than
    /// a right angle.
    pub draft: f64,
    /// The same angle in degrees.
    pub draft_degrees: f64,
    /// The bisector: a point on it (the first plane's point dropped onto it) and its unit normal,
    /// toward the first plane's side.
    pub bisector: OutwardPlane,
    /// A frame in the bisector: the unit direction of the line the two planes meet in, and the
    /// unit direction at right angles to it in the bisector, so that `x` cross `y` is the
    /// bisector's normal.
    pub x: Vec3,
    /// See `x`.
    pub y: Vec3,
}

fn check_planes(op: &'static str, a: &OutwardPlane, b: &OutwardPlane) -> Result<(Vec3, Vec3)> {
    if !(finite(a.point) && finite(b.point) && finite(a.outward) && finite(b.outward)) {
        return Err(SolidError::InvalidInput {
            operation: op,
            reason: "a plane's point and normal must be finite",
        });
    }
    Ok((a.outward.unitize()?, b.outward.unitize()?))
}

/// The bisector of `a` and `b` when their outward normals point apart (the angle between the
/// planes under a right angle) and are tilted past the kernel's angular tolerance.
///
/// `Ok(None)` is an answer, not a failure: the planes are parallel within the angular tolerance
/// (see [`facing_planes`]), or their normals do not point apart. Whether material lies between
/// them is read with [`drafted_thickness`] at the points where the faces overlap.
///
/// # Errors
/// [`SolidError::InvalidInput`] for a point or normal that is not finite, or a normal of zero
/// length.
pub fn drafted_planes(
    kernel: &Kernel,
    a: &OutwardPlane,
    b: &OutwardPlane,
) -> Result<Option<DraftedPlanes>> {
    let (ua, ub) = check_planes(DRAFT_OP, a, b)?;
    let cross = ua.cross(ub);
    let sine = cross.length();
    if ua.dot(ub) >= 0.0 || sine <= det::sin(kernel.tolerances.ang_tol_rad()) {
        return Ok(None);
    }
    // `ua - ub` is at least sqrt 2 long here (the normals point apart), so both unitize.
    let m = (ua - ub).unitize()?;
    let x = cross.unitize()?;
    let y = m.cross(x);
    // The bisector `da = db`: `p . (ua - ub) = pa . ua - pb . ub`, offset `c` along `m`.
    let offset = (a.point.dot(ua) - b.point.dot(ub)) / (ua - ub).length();
    let point = a.point - (m * (a.point.dot(m) - offset));
    let draft = det::atan2(sine, -ua.dot(ub));
    Ok(Some(DraftedPlanes {
        draft,
        draft_degrees: draft * 180.0 / crate::math::PI,
        bisector: OutwardPlane { point, outward: m },
        x,
        y,
    }))
}

/// The thickness of the drafted wall between `a` and `b` at each of `points`: the length of the
/// bisector's normal line through the point between the two planes, positive where the point
/// lies behind both planes (in the wall), negative in front of both, zero on the line they meet
/// in. A sheet's corners on the bisector give the wall's thickness there.
///
/// # Errors
/// [`SolidError::InvalidInput`] for a plane or point that is not finite, a normal of zero length,
/// more than [`crate::surface::mid_sheet::MAX_POINTS`] points, or planes that are not drafted
/// ([`drafted_planes`] answers `None`).
pub fn drafted_thickness(
    kernel: &Kernel,
    a: &OutwardPlane,
    b: &OutwardPlane,
    points: &[Point3],
) -> Result<Vec<f64>> {
    const OP: &str = "surface::face_pair::drafted_thickness";
    if points.len() > crate::surface::mid_sheet::MAX_POINTS {
        return Err(SolidError::InvalidInput {
            operation: OP,
            reason: "too many points",
        });
    }
    if !points.iter().all(|p| finite(*p)) {
        return Err(SolidError::InvalidInput {
            operation: OP,
            reason: "every point must be finite",
        });
    }
    let Some(d) = drafted_planes(kernel, a, b)? else {
        return Err(SolidError::InvalidInput {
            operation: OP,
            reason: "the two planes are not drafted: parallel, or their normals do not point apart",
        });
    };
    let (ua, ub) = check_planes(OP, a, b)?;
    // `cos(draft / 2)`: the bisector's normal against the first plane's, above cos 45 degrees.
    let half_cos = d.bisector.outward.dot(ua);
    Ok(points
        .iter()
        .map(|p| -(p.distance_along(a.point, ua) + p.distance_along(b.point, ub)) / half_cos)
        .collect())
}

/// `plane` moved `by` behind itself, against its outward normal (a negative `by` moves it in
/// front): the face of a wall's material `by` in from that face.
///
/// # Errors
/// [`SolidError::InvalidInput`] for a plane or distance that is not finite, or a normal of zero
/// length.
pub fn inset(plane: &OutwardPlane, by: f64) -> Result<OutwardPlane> {
    if !(finite(plane.point) && finite(plane.outward) && by.is_finite()) {
        return Err(SolidError::InvalidInput {
            operation: "surface::face_pair::inset",
            reason: "the plane and the distance must be finite",
        });
    }
    let n = plane.outward.unitize()?;
    Ok(OutwardPlane {
        point: plane.point - (n * by),
        outward: n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::KernelOptions;

    fn kernel() -> Kernel {
        Kernel::new(&KernelOptions::default())
    }

    fn plane(p: [f64; 3], n: [f64; 3]) -> OutwardPlane {
        OutwardPlane {
            point: Vec3::new(p[0], p[1], p[2]),
            outward: Vec3::new(n[0], n[1], n[2]),
        }
    }

    #[test]
    fn a_plates_two_faces_are_its_thickness_apart_with_the_mid_plane_between() {
        // A plate 2 thick: top at z = 2 facing up, bottom at z = 0 facing down. The points need
        // not be above one another.
        let top = plane([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]);
        let bottom = plane([5.0, -3.0, 0.0], [0.0, 0.0, -1.0]);
        let pair = facing_planes(&kernel(), &top, &bottom).unwrap().unwrap();
        assert!((pair.thickness - 2.0).abs() < 1e-12, "{pair:?}");
        assert!((pair.mid_point.z - 1.0).abs() < 1e-12, "{pair:?}");
        assert!((pair.toward_first.z - 1.0).abs() < 1e-12);
        // The other way round: the same wall, the mid plane at the same place.
        let back = facing_planes(&kernel(), &bottom, &top).unwrap().unwrap();
        assert!((back.thickness - 2.0).abs() < 1e-12);
        assert!((back.mid_point.z - 1.0).abs() < 1e-12);
        assert!((back.toward_first.z + 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_answer_does_not_depend_on_the_normals_length_or_the_frame() {
        // A wall 3 thick across the diagonal (1, 1, 0)/sqrt 2, normals not unit.
        let s = 0.5_f64.sqrt();
        let a = plane([3.0 * s, 3.0 * s, 0.0], [10.0 * s, 10.0 * s, 0.0]);
        let b = plane([0.0, 0.0, 7.0], [-s, -s, 0.0]);
        let pair = facing_planes(&kernel(), &a, &b).unwrap().unwrap();
        assert!((pair.thickness - 3.0).abs() < 1e-12, "{pair:?}");
        let mid_distance = pair.mid_point.dot(Vec3::new(s, s, 0.0));
        assert!((mid_distance - 1.5).abs() < 1e-12, "{pair:?}");
    }

    #[test]
    fn planes_that_bound_air_or_nothing_do_not_pair() {
        let k = kernel();
        // Normals pointing toward each other: a gap, not a wall.
        let low = plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let high = plane([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        assert_eq!(facing_planes(&k, &low, &high).unwrap(), None);
        // Opposite and at the same place: nothing between them.
        let same_a = plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        let same_b = plane([4.0, 4.0, 1.0], [0.0, 0.0, -1.0]);
        assert_eq!(facing_planes(&k, &same_a, &same_b).unwrap(), None);
        // Just past the zone tolerance apart pairs; at it does not.
        let zone = k.tolerances.zone_tol_3d;
        let thin = plane([0.0, 0.0, 2.0 * zone], [0.0, 0.0, 1.0]);
        let floor = plane([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(facing_planes(&k, &thin, &floor).unwrap().is_some());
        let at_zone = plane([0.0, 0.0, zone], [0.0, 0.0, 1.0]);
        assert_eq!(facing_planes(&k, &at_zone, &floor).unwrap(), None);
    }

    #[test]
    fn a_tilt_within_the_angular_tolerance_pairs_and_one_past_it_does_not() {
        let k = kernel();
        let top = plane([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]);
        let tilted = |deg: f64| {
            let t = deg.to_radians();
            plane([0.0, 0.0, 0.0], [det::sin(t), 0.0, -det::cos(t)])
        };
        // The tolerance is 0.5 degrees.
        assert!(facing_planes(&k, &top, &tilted(0.4)).unwrap().is_some());
        assert_eq!(facing_planes(&k, &top, &tilted(0.6)).unwrap(), None);
        // Along the same normals (not opposite) never pairs.
        assert_eq!(
            facing_planes(&k, &top, &plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])).unwrap(),
            None
        );
        // Perpendicular walls (a rim and a face) never pair.
        assert_eq!(
            facing_planes(&k, &top, &plane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])).unwrap(),
            None
        );
    }

    #[test]
    fn a_tilted_second_plane_gives_one_thickness_whichever_point_represents_it() {
        // `a` is z = 2 facing up; `b` passes through the origin with normal (sin t, 0, -cos t),
        // t = 0.4 degrees (within the tolerance). The normal line through (0, 0, 2) is the z axis,
        // which meets `b` at the origin: thickness 2, mid point (0, 0, 1). The point
        // (1000, 0, 1000 tan t) is on `b` too, (1000 sin t - 1000 tan t cos t = 0), and must give
        // the same answer.
        let k = kernel();
        let t = 0.4_f64.to_radians();
        let top = plane([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]);
        let n = [det::sin(t), 0.0, -det::cos(t)];
        for p in [[0.0, 0.0, 0.0], [1000.0, 0.0, 1000.0 * det::tan(t)]] {
            let pair = facing_planes(&k, &top, &plane(p, n)).unwrap().unwrap();
            assert!((pair.thickness - 2.0).abs() < 1e-9, "{p:?}: {pair:?}");
            assert!(
                (pair.mid_point - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-9,
                "{pair:?}"
            );
        }
    }

    #[test]
    fn hostile_input_is_refused_not_answered() {
        let k = kernel();
        let good = plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        for bad in [
            plane([f64::NAN, 0.0, 0.0], [0.0, 0.0, -1.0]),
            plane([0.0, f64::INFINITY, 0.0], [0.0, 0.0, -1.0]),
            plane([0.0, 0.0, 0.0], [0.0, f64::NAN, -1.0]),
            plane([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ] {
            assert!(facing_planes(&k, &good, &bad).is_err(), "{bad:?}");
            assert!(facing_planes(&k, &bad, &good).is_err(), "{bad:?}");
        }
    }

    /// A wedge: the bottom face z = 0 (facing down) and the top face through (0, 0, 2) tilted
    /// `deg` about the y axis, rising with x (`z = 2 + x tan deg`, facing up and back).
    fn wedge(deg: f64) -> (OutwardPlane, OutwardPlane) {
        let t = deg.to_radians();
        let top = plane([0.0, 0.0, 2.0], [-det::sin(t), 0.0, det::cos(t)]);
        let bottom = plane([7.0, 3.0, 0.0], [0.0, 0.0, -1.0]);
        (top, bottom)
    }

    #[test]
    fn a_wedges_bisector_halves_the_draft_through_the_line_the_faces_meet_in() {
        let k = kernel();
        let deg = 4.0_f64;
        let (top, bottom) = wedge(deg);
        let d = drafted_planes(&k, &top, &bottom).unwrap().unwrap();
        assert!((d.draft_degrees - deg).abs() < 1e-12, "{d:?}");
        assert!((d.draft - deg.to_radians()).abs() < 1e-12);
        // The bisector is tilted half the draft, its normal toward the top face.
        let half = (deg * 0.5).to_radians();
        let m = d.bisector.outward;
        assert!(
            (m - Vec3::new(-det::sin(half), 0.0, det::cos(half))).length() < 1e-12,
            "{m:?}"
        );
        // The faces meet on the line z = 0, x = -2 / tan(deg); the bisector passes through it.
        let line = Vec3::new(-2.0 / det::tan(deg.to_radians()), 0.0, 0.0);
        assert!(line.distance_along(d.bisector.point, m).abs() < 1e-9);
        // The frame: x along the line (the y axis), y in the bisector, x cross y the normal.
        assert!((d.x.cross(d.y) - m).length() < 1e-12);
        assert!(d.x.x.abs() < 1e-12 && d.x.z.abs() < 1e-12 && (d.x.y.abs() - 1.0).abs() < 1e-12);
        assert!(d.y.dot(m).abs() < 1e-12);
        // `s` along the bisector from the line, the wall is 2 s tan(draft / 2) thick.
        let along = Vec3::new(det::cos(half), 0.0, det::sin(half));
        let s = [5.0, 28.6, 100.0];
        let points: Vec<Vec3> = s.iter().map(|s| line + along * *s).collect();
        let t = drafted_thickness(&k, &top, &bottom, &points).unwrap();
        for (s, t) in s.iter().zip(&t) {
            let expected = 2.0 * s * det::tan(half);
            assert!(
                (t - expected).abs() < 1e-9,
                "at {s}: {t} against {expected}"
            );
        }
        // A point off the bisector, (0, 0, 1): the chord along the bisector's normal through it
        // is the sum of its depths behind the two planes over cos(draft / 2).
        let p = Vec3::new(0.0, 0.0, 1.0);
        let da = det::cos(deg.to_radians()) * (1.0 - 2.0);
        let db = -1.0;
        let t = drafted_thickness(&k, &top, &bottom, &[p]).unwrap();
        assert!((t[0] + (da + db) / det::cos(half)).abs() < 1e-12, "{t:?}");
        // Either face first: the same bisector, its normal toward the other face.
        let back = drafted_planes(&k, &bottom, &top).unwrap().unwrap();
        assert!((back.bisector.outward + m).length() < 1e-12);
        assert!(line.distance_along(back.bisector.point, m).abs() < 1e-9);
        let t_back = drafted_thickness(&k, &bottom, &top, &points).unwrap();
        let t_fwd = drafted_thickness(&k, &top, &bottom, &points).unwrap();
        for (x, y) in t_fwd.iter().zip(&t_back) {
            assert!((x - y).abs() < 1e-9, "{x} against {y}");
        }
    }

    #[test]
    fn past_the_line_the_faces_meet_in_and_across_air_the_thickness_is_negative() {
        let k = kernel();
        let (top, bottom) = wedge(3.0);
        let line = Vec3::new(-2.0 / det::tan(3.0_f64.to_radians()), 0.0, 0.0);
        let beyond = line - Vec3::new(10.0, 0.0, 0.0);
        let on_line = drafted_thickness(&k, &top, &bottom, &[line, beyond]).unwrap();
        assert!(on_line[0].abs() < 1e-9, "{on_line:?}");
        assert!(on_line[1] < 0.0, "{on_line:?}");
        // Normals toward each other across a gap: a bisector, but no wall anywhere near.
        let t = 3.0_f64.to_radians();
        let floor = plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let roof = plane([0.0, 0.0, 2.0], [det::sin(t), 0.0, -det::cos(t)]);
        assert!(drafted_planes(&k, &floor, &roof).unwrap().is_some());
        let gap = drafted_thickness(&k, &floor, &roof, &[Vec3::new(0.0, 0.0, 1.0)]).unwrap();
        assert!(gap[0] < 0.0, "{gap:?}");
    }

    #[test]
    fn parallel_or_same_way_planes_are_not_drafted() {
        let k = kernel();
        // Parallel within the angular tolerance (0.5 degrees): `facing_planes` pairs them.
        let (top, bottom) = wedge(0.4);
        assert_eq!(drafted_planes(&k, &top, &bottom).unwrap(), None);
        assert!(drafted_thickness(&k, &top, &bottom, &[Vec3::new(0.0, 0.0, 1.0)]).is_err());
        let (top, bottom) = wedge(0.6);
        assert!(drafted_planes(&k, &top, &bottom).unwrap().is_some());
        // Normals the same way, or at right angles: no wall between them.
        let up = plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let tilted_up = plane([0.0, 0.0, 2.0], [0.1, 0.0, 1.0]);
        assert_eq!(drafted_planes(&k, &up, &tilted_up).unwrap(), None);
        let side = plane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(drafted_planes(&k, &up, &side).unwrap(), None);
    }

    #[test]
    fn a_plane_moves_behind_itself_by_the_inset() {
        let p = inset(&plane([1.0, 2.0, 3.0], [0.0, 0.0, 5.0]), 0.25).unwrap();
        assert!((p.point - Vec3::new(1.0, 2.0, 2.75)).length() < 1e-12);
        assert!((p.outward - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-12);
        assert!(inset(&plane([0.0; 3], [0.0; 3]), 1.0).is_err());
        assert!(inset(&plane([0.0; 3], [0.0, 0.0, 1.0]), f64::NAN).is_err());
    }

    #[test]
    fn drafted_hostile_input_is_refused() {
        let k = kernel();
        let (top, bottom) = wedge(3.0);
        for bad in [
            plane([f64::NAN, 0.0, 0.0], [0.0, 0.0, -1.0]),
            plane([0.0, 0.0, 0.0], [0.0, f64::INFINITY, -1.0]),
            plane([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ] {
            assert!(drafted_planes(&k, &top, &bad).is_err(), "{bad:?}");
            assert!(drafted_planes(&k, &bad, &top).is_err(), "{bad:?}");
        }
        assert!(drafted_thickness(&k, &top, &bottom, &[Vec3::new(f64::NAN, 0.0, 0.0)]).is_err());
    }
}
