use crate::classifiers::hoeffding_tree::split_criteria::gini_split_criterion::GiniSplitCriterion;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;

pub trait SplitCriterion: Any {
    fn get_range_of_merit(&self, pre_split_distribution: &Vec<f64>) -> f64;
    fn get_merit_of_split(
        &self,
        pre_split_distribution: &[f64],
        post_split_dists: &[Vec<f64>],
    ) -> f64;
    fn as_any(&self) -> &dyn Any;
}

impl MemorySized for dyn SplitCriterion {
    fn inline_size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        if let Some(gini) = self.as_any().downcast_ref::<GiniSplitCriterion>() {
            gini.extra_heap_size(meter)
        } else {
            0
        }
    }
}
