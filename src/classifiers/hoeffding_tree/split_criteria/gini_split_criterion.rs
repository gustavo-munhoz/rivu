use crate::classifiers::hoeffding_tree::split_criteria::split_criterion::SplitCriterion;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::mem::size_of;

pub struct GiniSplitCriterion {}

impl GiniSplitCriterion {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compute_gini(&self, distribution: &Vec<f64>, distribution_sum_of_weights: f64) -> f64 {
        let mut gini = 1.0;
        for i in distribution {
            let rel_freq = i / distribution_sum_of_weights;
            gini -= rel_freq.powf(2.0);
        }
        gini
    }
}

impl SplitCriterion for GiniSplitCriterion {
    fn get_range_of_merit(&self, _pre_split_distribution: &Vec<f64>) -> f64 {
        1.0
    }

    fn get_merit_of_split(
        &self,
        _pre_split_distribution: &[f64],
        post_split_dists: &[Vec<f64>],
    ) -> f64 {
        let mut total_weight = 0.0;
        let mut dist_weights = Vec::with_capacity(post_split_dists.len());

        for dist in post_split_dists.iter() {
            let w = dist.iter().sum();
            dist_weights.push(w);
            total_weight += w;
        }

        let mut gini = 0.0;
        for (i, dist) in post_split_dists.iter().enumerate() {
            if total_weight > 0.0 {
                gini += (dist_weights[i] / total_weight) * self.compute_gini(dist, dist_weights[i]);
            }
        }

        1.0 - gini
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MemorySized for GiniSplitCriterion {
    fn inline_size(&self) -> usize {
        size_of::<Self>()
    }

    fn extra_heap_size(&self, _meter: &mut MemoryMeter) -> usize {
        0
    }
}
