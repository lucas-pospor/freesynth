//! Summary statistics.

use crate::report::Report;

/// Arithmetic mean (`NaN` for an empty slice).
pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Sample standard deviation (`n - 1` denominator; `NaN` for fewer than two values).
pub fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return f64::NAN;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// Standard error of the mean.
pub fn std_err(v: &[f64]) -> f64 {
    std_dev(v) / (v.len() as f64).sqrt()
}

/// Quantile `q` in `[0, 1]` of an ascending slice, interpolating linearly between order
/// statistics (type 7 of Hyndman and Fan, the default of R and NumPy).
///
/// # Panics
/// If `sorted` is empty.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty(), "quantile of an empty slice");
    let h = (sorted.len() - 1) as f64 * q.clamp(0.0, 1.0);
    let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Natural logarithm of the gamma function for `x > 0` (Lanczos approximation, g = 7, about 15
/// significant digits).
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let sum = G[1..].iter().enumerate().fold(G[0], |acc, (i, &g)| acc + g / (x + i as f64 + 1.0));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Regularized incomplete beta function `I_x(a, b)` (continued fraction, modified Lentz method;
/// Press et al., "Numerical Recipes", 3rd ed., section 6.4).
fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // The continued fraction converges fast for x < (a + 1) / (a + b + 2); use symmetry otherwise.
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - incomplete_beta(1.0 - x, b, a);
    }
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp() / a;
    const TINY: f64 = 1e-300;
    let (mut c, mut d) = (1.0f64, 1.0 - (a + b) * x / (a + 1.0));
    d = 1.0 / if d.abs() < TINY { TINY } else { d };
    let mut f = d;
    for m in 1..300 {
        let m = m as f64;
        for num in [
            m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0)),
        ] {
            d = 1.0 + num * d;
            d = 1.0 / if d.abs() < TINY { TINY } else { d };
            c = 1.0 + num / c;
            if c.abs() < TINY {
                c = TINY;
            }
            f *= c * d;
        }
        if (c * d - 1.0).abs() < 1e-15 {
            break;
        }
    }
    front * f
}

/// Cumulative distribution function of Student's t distribution with `df` degrees of freedom.
pub fn student_t_cdf(t: f64, df: f64) -> f64 {
    let tail = 0.5 * incomplete_beta(df / (df + t * t), df / 2.0, 0.5);
    if t >= 0.0 { 1.0 - tail } else { tail }
}

/// Quantile `q` in `(0, 1)` of Student's t distribution with `df > 0` degrees of freedom
/// (bisection on [`student_t_cdf`]).
pub fn student_t_quantile(q: f64, df: f64) -> f64 {
    assert!(q > 0.0 && q < 1.0 && df > 0.0, "t quantile needs q in (0, 1) and df > 0");
    if q < 0.5 {
        return -student_t_quantile(1.0 - q, df);
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    while student_t_cdf(hi, df) < q {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, df) < q {
            lo = mid
        } else {
            hi = mid
        }
        if hi - lo <= 1e-12 * hi {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Two-sided Student-t confidence interval of the mean of `v` (at least two values), assuming
/// independent, roughly normal values.
pub fn t_interval(v: &[f64], confidence: f64) -> Option<[f64; 2]> {
    if v.len() < 2 {
        return None;
    }
    let half = student_t_quantile(0.5 + confidence / 2.0, (v.len() - 1) as f64) * std_err(v);
    let m = mean(v);
    half.is_finite().then_some([m - half, m + half])
}

/// Sorts a copy of `v` ascending (`NaN`s last).
pub fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_unstable_by(f64::total_cmp);
    s
}

/// Spread of one metric over repeated runs.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricStats {
    pub metric: String,
    /// Number of reports that have the metric.
    pub n: usize,
    pub mean: f64,
    /// Sample standard deviation (`NaN` for one run).
    pub sd: f64,
    /// `sd / |mean|`.
    pub cv: f64,
    pub min: f64,
    pub max: f64,
}

/// Mean, standard deviation and range of every metric over `reports` (for example the same
/// synthesis with different seeds), in metric-name order.
pub fn metric_stats(reports: &[Report]) -> Vec<MetricStats> {
    let mut names: Vec<&String> = reports.iter().flat_map(|r| r.metrics.keys()).collect();
    names.sort();
    names.dedup();
    names
        .into_iter()
        .map(|m| {
            let v: Vec<f64> = reports.iter().filter_map(|r| r.metrics.get(m).copied()).collect();
            let (mean, sd) = (mean(&v), std_dev(&v));
            MetricStats {
                metric: m.clone(),
                n: v.len(),
                mean,
                sd,
                cv: sd / mean.abs(),
                min: v.iter().copied().fold(f64::INFINITY, f64::min),
                max: v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_over_reports() {
        let mut a = Report::new("energy");
        a.metric("total", 10.0).metric("style", 1.0);
        let mut b = Report::new("energy");
        b.metric("total", 12.0);
        let s = metric_stats(&[a, b]);
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].metric.as_str(), s[0].n), ("style", 1));
        assert!(s[0].sd.is_nan());
        assert_eq!((s[1].n, s[1].mean, s[1].min, s[1].max), (2, 11.0, 10.0, 12.0));
        assert!((s[1].sd - 2f64.sqrt()).abs() < 1e-12);
        assert!((s[1].cv - 2f64.sqrt() / 11.0).abs() < 1e-12);
    }

    #[test]
    fn student_t() {
        // Reference quantiles (R: qt).
        for (q, df, want) in [
            (0.975, 1.0, 12.706_204_736),
            (0.975, 4.0, 2.776_445_105),
            (0.975, 14.0, 2.144_786_688),
            (0.95, 30.0, 1.697_260_887),
            (0.995, 9.0, 3.249_835_542),
        ] {
            let got = student_t_quantile(q, df);
            assert!((got - want).abs() < 1e-6 * want, "t({q}, {df}) = {got}, expected {want}");
            assert!((student_t_quantile(1.0 - q, df) + want).abs() < 1e-6 * want);
        }
        assert!((student_t_cdf(0.0, 3.0) - 0.5).abs() < 1e-15);
        // Large df approaches the normal distribution.
        assert!((student_t_quantile(0.975, 1e6) - 1.959_964).abs() < 1e-4);
        let [lo, hi] = t_interval(&[1.0, 2.0, 3.0, 4.0], 0.95).unwrap();
        // mean 2.5, se 0.645497, t(0.975, 3) = 3.182446
        assert!((lo - (2.5 - 3.182_446_305 * 0.645_497_224)).abs() < 1e-6 && (hi + lo - 5.0).abs() < 1e-9);
        assert!(t_interval(&[1.0], 0.95).is_none());
    }

    #[test]
    fn basics() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mean(&v), 2.5);
        assert!((std_dev(&v) - 1.290_994_448_7).abs() < 1e-9);
        assert!((std_err(&v) - 0.645_497_224_4).abs() < 1e-9);
        assert_eq!(quantile_sorted(&v, 0.0), 1.0);
        assert_eq!(quantile_sorted(&v, 1.0), 4.0);
        assert_eq!(quantile_sorted(&v, 0.5), 2.5);
        assert!((quantile_sorted(&v, 0.9) - 3.7).abs() < 1e-12);
        assert!(std_dev(&[1.0]).is_nan());
        assert_eq!(sorted(&[3.0, 1.0, 2.0]), vec![1.0, 2.0, 3.0]);
    }
}
