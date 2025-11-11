use crate::classifiers::attribute_class_observers::AttributeClassObserver;
use crate::classifiers::conditional_tests::attribute_split_suggestion::AttributeSplitSuggestion;
use crate::classifiers::hoeffding_tree::split_criteria::SplitCriterion;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::mem::size_of;

pub struct NullAttributeClassObserver {}

impl NullAttributeClassObserver {
    pub fn new() -> Self {
        NullAttributeClassObserver {}
    }
}

impl AttributeClassObserver for NullAttributeClassObserver {
    fn observe_attribute_class(&mut self, _att_val: f64, _class_val: usize, _weight: f64) {}

    fn probability_of_attribute_value_given_class(
        &self,
        _att_val: f64,
        _class_val: usize,
    ) -> Option<f64> {
        Some(0.0)
    }

    fn get_best_evaluated_split_suggestion(
        &self,
        _criterion: &dyn SplitCriterion,
        _pre_split_dist: &[f64],
        _att_index: usize,
        _binary_only: bool,
    ) -> Option<AttributeSplitSuggestion> {
        None
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

impl MemorySized for NullAttributeClassObserver {
    fn inline_size(&self) -> usize {
        size_of::<Self>()
    }
}
