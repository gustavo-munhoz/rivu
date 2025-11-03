use crate::classifiers::attribute_class_observers::AttributeClassObserver;
use crate::classifiers::conditional_tests::attribute_split_suggestion::AttributeSplitSuggestion;
use crate::classifiers::hoeffding_tree::split_criteria::SplitCriterion;
use std::any::Any;

pub struct NullAttributeClassObserver {}

impl NullAttributeClassObserver {
    pub fn new() -> Self {
        NullAttributeClassObserver {}
    }
}

impl AttributeClassObserver for NullAttributeClassObserver {
    fn observe_attribute_class(&mut self, att_val: f64, class_val: usize, weight: f64) {}

    fn probability_of_attribute_value_given_class(
        &self,
        att_val: f64,
        class_val: usize,
    ) -> Option<f64> {
        Some(0.0)
    }

    fn get_best_evaluated_split_suggestion(
        &self,
        criterion: &dyn SplitCriterion,
        pre_split_dist: &[f64],
        att_index: usize,
        binary_only: bool,
    ) -> Option<AttributeSplitSuggestion> {
        None
    }

    fn calc_memory_size(&self) -> usize {
        size_of::<Self>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
