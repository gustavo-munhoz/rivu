use crate::classifiers::attribute_class_observers::attribute_class_observer::AttributeClassObserver;
use crate::classifiers::conditional_tests::attribute_split_suggestion::AttributeSplitSuggestion;
use crate::classifiers::hoeffding_tree::instance_conditional_test::{
    NominalAttributeBinaryTest, NominalAttributeMultiwayTest,
};
use crate::classifiers::hoeffding_tree::split_criteria::SplitCriterion;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::mem::size_of;

pub struct NominalAttributeClassObserver {
    total_weight_observed: f64,
    missing_weight_observed: f64,
    attribute_value_distribution_per_class: Vec<Vec<f64>>,
}

impl NominalAttributeClassObserver {
    pub fn new() -> NominalAttributeClassObserver {
        NominalAttributeClassObserver {
            total_weight_observed: 0.0,
            missing_weight_observed: 0.0,
            attribute_value_distribution_per_class: Vec::new(),
        }
    }

    #[inline]
    fn ensure_class(&mut self, class_val: usize) {
        if class_val >= self.attribute_value_distribution_per_class.len() {
            self.attribute_value_distribution_per_class
                .resize_with(class_val + 1, Vec::new);
        }
    }

    #[inline]
    fn ensure_value(&mut self, class_val: usize, att_val_int: usize) {
        self.ensure_class(class_val);
        let row = &mut self.attribute_value_distribution_per_class[class_val];
        if att_val_int >= row.len() {
            row.resize(att_val_int + 1, 0.0);
        }
    }

    pub fn get_max_att_vals_observed(&self) -> usize {
        self.attribute_value_distribution_per_class
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0)
    }

    pub fn get_class_dists_resulting_from_multiway_split(
        &self,
        max_att_vals: usize,
    ) -> Vec<Vec<f64>> {
        let mut dists =
            vec![vec![0.0; self.attribute_value_distribution_per_class.len()]; max_att_vals];

        for (class_idx, row) in self
            .attribute_value_distribution_per_class
            .iter()
            .enumerate()
        {
            for (val_idx, &count) in row.iter().enumerate() {
                dists[val_idx][class_idx] = count;
            }
        }
        dists
    }

    pub fn get_class_resulting_from_binary_split(&self, val_index: usize) -> Vec<Vec<f64>> {
        let num_classes = self.attribute_value_distribution_per_class.len();
        let mut lhs = vec![0.0; num_classes];
        let mut rhs = vec![0.0; num_classes];

        for (class_idx, row) in self
            .attribute_value_distribution_per_class
            .iter()
            .enumerate()
        {
            let lhs_count = *row.get(val_index).unwrap_or(&0.0);
            lhs[class_idx] += lhs_count;
            let rhs_count: f64 = row.iter().copied().sum::<f64>() - lhs_count;
            rhs[class_idx] += rhs_count;
        }
        vec![lhs, rhs]
    }
}

impl AttributeClassObserver for NominalAttributeClassObserver {
    fn observe_attribute_class(&mut self, att_val: f64, class_val: usize, weight: f64) {
        if att_val.is_nan() {
            self.missing_weight_observed += weight;
        } else {
            let att_val_int = att_val as usize;
            self.ensure_value(class_val, att_val_int);
            self.attribute_value_distribution_per_class[class_val][att_val_int] += weight;
        }
        self.total_weight_observed += weight;
    }

    fn probability_of_attribute_value_given_class(
        &self,
        att_val: f64,
        class_val: usize,
    ) -> Option<f64> {
        if att_val.is_nan() {
            return None;
        }
        let att_val_int = att_val as usize;
        let row = self.attribute_value_distribution_per_class.get(class_val)?;
        if row.is_empty() {
            return None;
        }
        let count = row.get(att_val_int).copied().unwrap_or(0.0);
        let sum: f64 = row.iter().copied().sum();
        let k = row.len() as f64;
        Some((count + 1.0) / (sum + k))
    }

    fn get_best_evaluated_split_suggestion(
        &self,
        criterion: &dyn SplitCriterion,
        pre_split_dist: &[f64],
        att_index: usize,
        binary_only: bool,
    ) -> Option<AttributeSplitSuggestion> {
        let mut best: Option<AttributeSplitSuggestion> = None;
        let max_att_vals_observed = self.get_max_att_vals_observed();

        if !binary_only {
            let post_split_dists =
                self.get_class_dists_resulting_from_multiway_split(max_att_vals_observed);
            let merit = criterion.get_merit_of_split(pre_split_dist, &post_split_dists);

            best = Some(AttributeSplitSuggestion::new(
                Some(Box::new(NominalAttributeMultiwayTest::new(att_index))),
                post_split_dists,
                merit,
            ));
        }

        for val_index in 0..max_att_vals_observed {
            let post_split_dists = self.get_class_resulting_from_binary_split(val_index);
            let merit = criterion.get_merit_of_split(pre_split_dist, &post_split_dists);

            if best.is_none() || merit > best.as_ref().unwrap().get_merit() {
                best = Some(AttributeSplitSuggestion::new(
                    Some(Box::new(NominalAttributeBinaryTest::new(
                        att_index, val_index,
                    ))),
                    post_split_dists,
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

impl MemorySized for NominalAttributeClassObserver {
    fn inline_size(&self) -> usize {
        size_of::<Self>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        meter.measure_field(&self.attribute_value_distribution_per_class)
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
    fn starts_empty() {
        let obs = NominalAttributeClassObserver::new();
        assert!(
            obs.probability_of_attribute_value_given_class(0.0, 0)
                .is_none()
        );
        assert!(approx_eq(obs.total_weight_observed, 0.0, EPS));
        assert!(approx_eq(obs.missing_weight_observed, 0.0, EPS));
        assert!(obs.attribute_value_distribution_per_class.is_empty());
    }

    #[test]
    fn laplace_probabilities_simple_case() {
        let mut obs = NominalAttributeClassObserver::new();
        obs.observe_attribute_class(0.0, 0, 1.0);
        obs.observe_attribute_class(0.0, 0, 1.0);
        obs.observe_attribute_class(0.0, 0, 1.0);
        obs.observe_attribute_class(1.0, 0, 1.0);

        let p0 = obs
            .probability_of_attribute_value_given_class(0.0, 0)
            .unwrap();
        let p1 = obs
            .probability_of_attribute_value_given_class(1.0, 0)
            .unwrap();
        assert!(approx_eq(p0, 4.0 / 6.0, 1e-12));
        assert!(approx_eq(p1, 2.0 / 6.0, 1e-12));

        let row = &obs.attribute_value_distribution_per_class[0];
        let mut sum_probs = 0.0;
        for (val_idx, _) in row.iter().enumerate() {
            let p = obs
                .probability_of_attribute_value_given_class(val_idx as f64, 0)
                .unwrap();
            sum_probs += p;
        }
        assert!(approx_eq(sum_probs, 1.0, 1e-12));

        assert!(approx_eq(obs.total_weight_observed, 4.0, 1e-12));
        assert!(approx_eq(obs.missing_weight_observed, 0.0, 1e-12));
    }

    #[test]
    fn different_classes_are_independent() {
        let mut obs = NominalAttributeClassObserver::new();
        obs.observe_attribute_class(0.0, 0, 1.0);
        obs.observe_attribute_class(0.0, 0, 1.0);
        obs.observe_attribute_class(1.0, 0, 1.0);
        obs.observe_attribute_class(1.0, 1, 1.0);
        obs.observe_attribute_class(1.0, 1, 1.0);

        let p0_c0 = obs
            .probability_of_attribute_value_given_class(0.0, 0)
            .unwrap(); // (2+1)/(3+2)=3/5=0.6
        let p1_c0 = obs
            .probability_of_attribute_value_given_class(1.0, 0)
            .unwrap(); // (1+1)/5=0.4
        assert!(approx_eq(p0_c0, 0.6, 1e-12));
        assert!(approx_eq(p1_c0, 0.4, 1e-12));

        let p0_c1 = obs
            .probability_of_attribute_value_given_class(0.0, 1)
            .unwrap(); // (0+1)/(2+2)=0.25
        let p1_c1 = obs
            .probability_of_attribute_value_given_class(1.0, 1)
            .unwrap(); // (2+1)/4=0.75
        assert!(approx_eq(p0_c1, 0.25, 1e-12));
        assert!(approx_eq(p1_c1, 0.75, 1e-12));
    }

    #[test]
    fn handles_missing_values_and_weights() {
        let mut obs = NominalAttributeClassObserver::new();
        obs.observe_attribute_class(f64::NAN, 0, 2.5);
        obs.observe_attribute_class(2.0, 0, 1.5);

        assert!(approx_eq(obs.missing_weight_observed, 2.5, 1e-12));
        assert!(approx_eq(obs.total_weight_observed, 4.0, 1e-12));
        assert_eq!(obs.attribute_value_distribution_per_class.len(), 1);

        let p = obs
            .probability_of_attribute_value_given_class(2.0, 0)
            .unwrap();
        assert!(approx_eq(p, 2.5 / 4.5, 1e-12));

        let p0 = obs
            .probability_of_attribute_value_given_class(0.0, 0)
            .unwrap();
        let p1 = obs
            .probability_of_attribute_value_given_class(1.0, 0)
            .unwrap();
        assert!(approx_eq(p0, 1.0 / 4.5, 1e-12));
        assert!(approx_eq(p1, 1.0 / 4.5, 1e-12));

        let row = &obs.attribute_value_distribution_per_class[0];
        let mut sum_probs = 0.0;
        for (val_idx, _) in row.iter().enumerate() {
            sum_probs += obs
                .probability_of_attribute_value_given_class(val_idx as f64, 0)
                .unwrap();
        }
        assert!(approx_eq(sum_probs, 1.0, 1e-12));
    }

    #[test]
    fn returns_none_when_class_exists_but_row_empty() {
        let mut obs = NominalAttributeClassObserver::new();
        obs.ensure_class(3);
        assert!(
            obs.probability_of_attribute_value_given_class(0.0, 3)
                .is_none()
        );
    }

    #[test]
    fn probability_none_for_out_of_bounds_class() {
        let obs = NominalAttributeClassObserver::new();
        assert!(
            obs.probability_of_attribute_value_given_class(0.0, 10)
                .is_none()
        );
    }

    #[test]
    fn large_value_index_expands_row() {
        let mut obs = NominalAttributeClassObserver::new();
        obs.observe_attribute_class(7.0, 0, 2.0);

        assert!(obs.attribute_value_distribution_per_class[0].len() >= 8);

        let p7 = obs
            .probability_of_attribute_value_given_class(7.0, 0)
            .unwrap();
        assert!(approx_eq(p7, 3.0 / 10.0, 1e-12));

        let p0 = obs
            .probability_of_attribute_value_given_class(0.0, 0)
            .unwrap();
        assert!(approx_eq(p0, 1.0 / 10.0, 1e-12));

        let row = &obs.attribute_value_distribution_per_class[0];
        let sum: f64 = (0..row.len())
            .map(|i| {
                obs.probability_of_attribute_value_given_class(i as f64, 0)
                    .unwrap()
            })
            .sum();
        assert!(approx_eq(sum, 1.0, 1e-12));
    }
}
