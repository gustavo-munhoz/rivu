use crate::utils::math::normal_probability;

#[derive(Clone, Debug, Default)]
pub struct GaussianEstimator {
    weight_sum: f64,
    mean: f64,
    variance_sum: f64,
}

impl GaussianEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn add_observation(&mut self, value: f64, weight: f64) {
        if value.is_infinite() || value.is_nan() {
            return;
        }

        if self.weight_sum > 0.0 {
            self.weight_sum += weight;
            let last_mean = self.mean;
            self.mean += weight * (value - last_mean) / self.weight_sum;
            self.variance_sum += weight * (value - last_mean) * (value - self.mean);
        } else {
            self.mean = value;
            self.weight_sum = weight;
        }
    }

    pub fn get_variance(&self) -> f64 {
        if self.weight_sum > 1.0 {
            self.variance_sum / (self.weight_sum - 1.0)
        } else {
            0.0
        }
    }

    pub fn get_std_dev(&self) -> f64 {
        self.get_variance().sqrt()
    }

    pub fn get_total_weight_observed(&self) -> f64 {
        self.weight_sum
    }

    pub fn estimated_weight_less_equal_greater_value(&self, value: f64) -> [f64; 3] {
        let equal_weight = self.probability_density(value) * self.weight_sum;
        let std_dev = self.get_std_dev();
        let less_weight = if std_dev > 0.0 {
            let z = (value - self.mean) / std_dev;
            normal_probability(z) * self.weight_sum - equal_weight
        } else {
            if value < self.mean {
                self.weight_sum - equal_weight
            } else {
                0.0
            }
        };

        let mut greater_weight = self.weight_sum - equal_weight - less_weight;
        if greater_weight < 0.0 {
            greater_weight = 0.0;
        }
        [less_weight, equal_weight, greater_weight]
    }

    #[inline]
    pub fn add_observations(&mut self, observer: &GaussianEstimator) {
        if (self.weight_sum > 0.0) && (observer.weight_sum > 0.0) {
            let old_mean = self.mean;
            self.mean = (self.mean * (self.weight_sum / (self.weight_sum + observer.weight_sum)))
                + (observer.mean * (observer.weight_sum / (self.weight_sum + observer.weight_sum)));
            self.variance_sum += observer.variance_sum
                + (self.weight_sum * observer.weight_sum / (self.weight_sum + observer.weight_sum)
                    * (observer.mean - old_mean).powi(2));
            self.weight_sum += observer.weight_sum;
        }
    }

    pub fn probability_density(&self, value: f64) -> f64 {
        let normal_const: f64 = (2.0 * std::f64::consts::PI).sqrt();
        if self.weight_sum > 0.0 {
            let std_dev = self.get_std_dev();
            if std_dev > 0.0 {
                let diff = value - self.mean;
                return (1.0 / (normal_const * std_dev))
                    * ((-diff * diff) / (2.0 * std_dev * std_dev)).exp();
            }
            return if (value - self.mean).abs() == 0.0 {
                1.0
            } else {
                0.0
            };
        }
        0.0
    }

    pub fn calc_memory_size(&self) -> usize {
        size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::GaussianEstimator;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn starts_empty() {
        let g = GaussianEstimator::new();
        assert!(approx_eq(g.get_variance(), 0.0, EPS));
        assert!(approx_eq(g.get_std_dev(), 0.0, EPS));
        assert!(approx_eq(g.probability_density(0.0), 0.0, EPS));
        assert!(approx_eq(g.weight_sum, 0.0, EPS));
    }

    #[test]
    fn single_observation_zero_variance() {
        let mut g = GaussianEstimator::new();
        g.add_observation(5.0, 1.0);

        assert!(approx_eq(g.mean, 5.0, EPS));
        assert!(approx_eq(g.weight_sum, 1.0, EPS));
        assert!(approx_eq(g.get_variance(), 0.0, EPS));
        assert!(approx_eq(g.get_std_dev(), 0.0, EPS));

        assert!(approx_eq(g.probability_density(5.0), 1.0, 1e-12));
        assert!(approx_eq(g.probability_density(4.999999), 0.0, 1e-12));
    }

    #[test]
    fn two_observations_sample_variance_matches() {
        let mut g = GaussianEstimator::new();
        g.add_observation(0.0, 1.0);
        g.add_observation(2.0, 1.0);

        assert!(approx_eq(g.mean, 1.0, EPS));
        assert!(approx_eq(g.weight_sum, 2.0, EPS));

        assert!(approx_eq(g.get_variance(), 2.0, 1e-12));
        assert!(approx_eq(g.get_std_dev(), (2.0f64).sqrt(), 1e-12));
    }

    #[test]
    fn three_observations_variance_known() {
        let mut g = GaussianEstimator::new();
        g.add_observation(-1.0, 1.0);
        g.add_observation(0.0, 1.0);
        g.add_observation(1.0, 1.0);

        assert!(approx_eq(g.mean, 0.0, EPS));
        assert!(approx_eq(g.weight_sum, 3.0, EPS));
        assert!(approx_eq(g.get_variance(), 1.0, 1e-12));
        assert!(approx_eq(g.get_std_dev(), 1.0, 1e-12));

        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!(approx_eq(g.probability_density(0.0), expected, 1e-9));
    }

    #[test]
    fn weighted_data_behaves_like_repetition() {
        let mut gw = GaussianEstimator::new();
        gw.add_observation(0.0, 2.0);
        gw.add_observation(2.0, 1.0);

        let mut g_rep = GaussianEstimator::new();
        g_rep.add_observation(0.0, 1.0);
        g_rep.add_observation(0.0, 1.0);
        g_rep.add_observation(2.0, 1.0);

        assert!(approx_eq(gw.mean, g_rep.mean, 1e-12));
        assert!(approx_eq(gw.get_variance(), g_rep.get_variance(), 1e-12));
        assert!(approx_eq(gw.weight_sum, g_rep.weight_sum, 1e-12));
    }

    #[test]
    fn add_observations_combines_correctly() {
        let mut a = GaussianEstimator::new();
        a.add_observation(0.0, 1.0);
        a.add_observation(2.0, 1.0);

        let mut b = GaussianEstimator::new();
        b.add_observation(4.0, 1.0);
        b.add_observation(6.0, 1.0);

        let mut c = GaussianEstimator::new();
        c.add_observation(0.0, 1.0);
        c.add_observation(2.0, 1.0);
        c.add_observation(4.0, 1.0);
        c.add_observation(6.0, 1.0);

        let mut combined = a.clone();
        combined.add_observations(&b);

        assert!(approx_eq(combined.mean, c.mean, 1e-12));
        assert!(approx_eq(combined.get_variance(), c.get_variance(), 1e-12));
        assert!(approx_eq(combined.weight_sum, c.weight_sum, 1e-12));
    }

    #[test]
    fn ignores_invalid_values() {
        let mut g = GaussianEstimator::new();
        g.add_observation(f64::NAN, 1.0);
        g.add_observation(f64::INFINITY, 1.0);
        g.add_observation(f64::NEG_INFINITY, 1.0);

        assert!(approx_eq(g.weight_sum, 0.0, EPS));
        assert!(approx_eq(g.get_variance(), 0.0, EPS));
        assert!(approx_eq(g.probability_density(0.0), 0.0, EPS));
    }

    #[test]
    fn pdf_with_zero_variance_is_spike_at_mean() {
        let mut g = GaussianEstimator::new();
        g.add_observation(10.0, 3.0);
        assert!(approx_eq(g.get_std_dev(), 0.0, EPS));
        assert!(approx_eq(g.probability_density(10.0), 1.0, 1e-12));
        assert!(approx_eq(g.probability_density(9.999999999), 0.0, 1e-12));
    }
}
