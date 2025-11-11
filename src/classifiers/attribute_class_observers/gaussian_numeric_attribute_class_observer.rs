use crate::classifiers::attribute_class_observers::attribute_class_observer::AttributeClassObserver;
use crate::classifiers::conditional_tests::attribute_split_suggestion::AttributeSplitSuggestion;
use crate::classifiers::hoeffding_tree::instance_conditional_test::NumericAttributeBinaryTest;
use crate::classifiers::hoeffding_tree::split_criteria::SplitCriterion;
use crate::core::estimators::gaussian_estimator::GaussianEstimator;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::mem::size_of;
pub struct GaussianNumericAttributeClassObserver {
    min_value_observed_per_class: Vec<f64>,
    max_value_observed_per_class: Vec<f64>,
    attribute_value_distribution_per_class: Vec<Option<GaussianEstimator>>,
    num_bins_option: usize,
}

impl GaussianNumericAttributeClassObserver {
    pub fn new() -> Self {
        GaussianNumericAttributeClassObserver {
            min_value_observed_per_class: Vec::new(),
            max_value_observed_per_class: Vec::new(),
            attribute_value_distribution_per_class: Vec::new(),
            num_bins_option: 10,
        }
    }

    #[inline]
    fn ensure_class(&mut self, class_val: usize) {
        if class_val >= self.attribute_value_distribution_per_class.len() {
            let new_len = class_val + 1;
            self.attribute_value_distribution_per_class
                .resize_with(new_len, || None);

            self.min_value_observed_per_class
                .resize_with(new_len, || 0.0);
            self.max_value_observed_per_class
                .resize_with(new_len, || 0.0);
        }
    }

    fn get_split_point_suggestions(&self) -> Vec<f64> {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for (i, est_opt) in self
            .attribute_value_distribution_per_class
            .iter()
            .enumerate()
        {
            if est_opt.is_some() {
                if self.min_value_observed_per_class[i] < min_val {
                    min_val = self.min_value_observed_per_class[i];
                }
                if self.max_value_observed_per_class[i] > max_val {
                    max_val = self.max_value_observed_per_class[i];
                }
            }
        }

        if min_val == f64::INFINITY || max_val == f64::NEG_INFINITY {
            return vec![];
        }

        let range = max_val - min_val;
        let mut suggestions = Vec::new();

        for i in 0..self.num_bins_option {
            let split_value =
                (range / (self.num_bins_option as f64 + 1.0)) * (i as f64 + 1.0) + min_val;
            if split_value > min_val && split_value < max_val {
                suggestions.push(split_value);
            }
        }

        suggestions
    }

    fn get_class_dists_resulting_from_binary_split(&self, split_value: f64) -> Vec<Vec<f64>> {
        let num_classes = self.attribute_value_distribution_per_class.len();
        let mut lhs = vec![0.0; num_classes];
        let mut rhs = vec![0.0; num_classes];

        for (class_idx, est_opt) in self
            .attribute_value_distribution_per_class
            .iter()
            .enumerate()
        {
            if let Some(est) = est_opt {
                if split_value < self.min_value_observed_per_class[class_idx] {
                    rhs[class_idx] += est.get_total_weight_observed()
                } else if split_value >= self.max_value_observed_per_class[class_idx] {
                    lhs[class_idx] += est.get_total_weight_observed()
                } else {
                    let [less, equal, greater] =
                        est.estimated_weight_less_equal_greater_value(split_value);
                    lhs[class_idx] += less + equal;
                    rhs[class_idx] += greater;
                }
            }
        }
        vec![lhs, rhs]
    }
}

impl AttributeClassObserver for GaussianNumericAttributeClassObserver {
    fn observe_attribute_class(&mut self, att_val: f64, class_val: usize, weight: f64) {
        if att_val.is_nan() || !weight.is_finite() || weight <= 0.0 {
            return;
        }

        self.ensure_class(class_val);

        let val_dist = &mut self.attribute_value_distribution_per_class[class_val];
        if val_dist.is_none() {
            let mut new_est = GaussianEstimator::new();
            new_est.add_observation(att_val, weight);
            *val_dist = Some(new_est);
            self.min_value_observed_per_class[class_val] = att_val;
            self.max_value_observed_per_class[class_val] = att_val;
        } else {
            if att_val < self.min_value_observed_per_class[class_val] {
                self.min_value_observed_per_class[class_val] = att_val;
            }
            if att_val > self.max_value_observed_per_class[class_val] {
                self.max_value_observed_per_class[class_val] = att_val;
            }
            val_dist.as_mut().unwrap().add_observation(att_val, weight);
        }
    }

    fn probability_of_attribute_value_given_class(
        &self,
        att_val: f64,
        class_val: usize,
    ) -> Option<f64> {
        if att_val.is_nan() {
            return None;
        }
        match self.attribute_value_distribution_per_class.get(class_val) {
            Some(Some(est)) => {
                if est.get_total_weight_observed() <= 0.0 {
                    None
                } else {
                    Some(est.probability_density(att_val))
                }
            }
            _ => None,
        }
    }

    fn get_best_evaluated_split_suggestion(
        &self,
        criterion: &dyn SplitCriterion,
        pre_split_dist: &[f64],
        att_index: usize,
        _binary_only: bool,
    ) -> Option<AttributeSplitSuggestion> {
        let split_points = self.get_split_point_suggestions();
        let mut best: Option<AttributeSplitSuggestion> = None;

        for split_value in split_points {
            let post_dists = self.get_class_dists_resulting_from_binary_split(split_value);
            let merit = criterion.get_merit_of_split(pre_split_dist, &post_dists);

            if best.is_none() || merit > best.as_ref().unwrap().get_merit() {
                best = Some(AttributeSplitSuggestion::new(
                    Some(Box::new(NumericAttributeBinaryTest::new(
                        att_index,
                        split_value,
                        true,
                    ))),
                    post_dists,
                    merit,
                ));
            }
        }
        best
    }

    fn calc_memory_size(&self) -> usize {
        MemoryMeter::measure_root(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl MemorySized for GaussianNumericAttributeClassObserver {
    fn inline_size(&self) -> usize {
        size_of::<Self>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        let mut total = 0;
        total += meter.measure_field(&self.min_value_observed_per_class);
        total += meter.measure_field(&self.max_value_observed_per_class);
        total += meter.measure_field(&self.attribute_value_distribution_per_class);
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn starts_empty_returns_none() {
        let obs = GaussianNumericAttributeClassObserver::new();
        assert!(
            obs.probability_of_attribute_value_given_class(0.0, 0)
                .is_none()
        );
    }

    #[test]
    fn lazy_init_and_pdf_reasonable() {
        let mut obs = GaussianNumericAttributeClassObserver::new();

        obs.observe_attribute_class(1.0, 0, 1.0);
        obs.observe_attribute_class(3.0, 0, 1.0);
        obs.observe_attribute_class(2.0, 0, 1.0);

        let p_center = obs
            .probability_of_attribute_value_given_class(2.0, 0)
            .unwrap();
        let p_far1 = obs
            .probability_of_attribute_value_given_class(0.0, 0)
            .unwrap();
        let p_far2 = obs
            .probability_of_attribute_value_given_class(5.0, 0)
            .unwrap();
        assert!(p_center > p_far1);
        assert!(p_center > p_far2);

        assert!(
            obs.probability_of_attribute_value_given_class(2.0, 1)
                .is_none()
        );
    }

    #[test]
    fn updates_min_max_per_class() {
        let mut obs = GaussianNumericAttributeClassObserver::new();

        obs.observe_attribute_class(10.0, 2, 1.0);
        obs.observe_attribute_class(8.0, 2, 1.0);
        obs.observe_attribute_class(12.0, 2, 1.0);

        let p8 = obs
            .probability_of_attribute_value_given_class(8.0, 2)
            .unwrap();
        let p12 = obs
            .probability_of_attribute_value_given_class(12.0, 2)
            .unwrap();
        let p5 = obs
            .probability_of_attribute_value_given_class(5.0, 2)
            .unwrap();
        let p15 = obs
            .probability_of_attribute_value_given_class(15.0, 2)
            .unwrap();
        assert!(p8 > p5);
        assert!(p12 > p15);
    }

    #[test]
    fn ignores_nan_and_zero_weight() {
        let mut obs = GaussianNumericAttributeClassObserver::new();

        obs.observe_attribute_class(f64::NAN, 0, 1.0);
        assert!(
            obs.probability_of_attribute_value_given_class(0.0, 0)
                .is_none()
        );

        obs.observe_attribute_class(10.0, 0, 0.0);
        assert!(
            obs.probability_of_attribute_value_given_class(10.0, 0)
                .is_none()
        );

        obs.observe_attribute_class(10.0, 0, 2.0);
        let p = obs
            .probability_of_attribute_value_given_class(10.0, 0)
            .unwrap();
        assert!(approx_eq(p, 1.0, EPS));
        let p_off = obs
            .probability_of_attribute_value_given_class(9.999_999_999, 0)
            .unwrap();
        assert!(approx_eq(p_off, 0.0, EPS));
    }

    #[test]
    fn class_index_out_of_bounds_returns_none() {
        let mut obs = GaussianNumericAttributeClassObserver::new();
        obs.observe_attribute_class(1.0, 0, 1.0);
        assert!(
            obs.probability_of_attribute_value_given_class(1.0, 5)
                .is_none()
        );
    }

    #[test]
    fn pdf_monotonic_around_mean_for_simple_case() {
        let mut obs = GaussianNumericAttributeClassObserver::new();
        obs.observe_attribute_class(-1.0, 0, 1.0);
        obs.observe_attribute_class(0.0, 0, 1.0);
        obs.observe_attribute_class(1.0, 0, 1.0);

        let p0 = obs
            .probability_of_attribute_value_given_class(0.0, 0)
            .unwrap();
        let p1 = obs
            .probability_of_attribute_value_given_class(1.0, 0)
            .unwrap();
        let p2 = obs
            .probability_of_attribute_value_given_class(2.0, 0)
            .unwrap();

        assert!(p0 > p1);
        assert!(p1 > p2);
    }
}
