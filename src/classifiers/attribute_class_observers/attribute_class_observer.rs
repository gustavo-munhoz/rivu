use crate::classifiers::attribute_class_observers::gaussian_numeric_attribute_class_observer::GaussianNumericAttributeClassObserver;
use crate::classifiers::attribute_class_observers::nominal_attribute_class_observer::NominalAttributeClassObserver;
use crate::classifiers::attribute_class_observers::null_attribute_class_observer::NullAttributeClassObserver;
use crate::classifiers::conditional_tests::attribute_split_suggestion::AttributeSplitSuggestion;
use crate::classifiers::hoeffding_tree::split_criteria::SplitCriterion;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;

pub trait AttributeClassObserver {
    fn observe_attribute_class(&mut self, att_val: f64, class_val: usize, weight: f64);
    fn probability_of_attribute_value_given_class(
        &self,
        att_val: f64,
        class_val: usize,
    ) -> Option<f64>;
    fn get_best_evaluated_split_suggestion(
        &self,
        criterion: &dyn SplitCriterion,
        pre_split_dist: &[f64],
        att_index: usize,
        binary_only: bool,
    ) -> Option<AttributeSplitSuggestion>;
    fn calc_memory_size(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl MemorySized for dyn AttributeClassObserver {
    fn inline_size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        if let Some(nominal) = self
            .as_any()
            .downcast_ref::<NominalAttributeClassObserver>()
        {
            nominal.extra_heap_size(meter)
        } else if let Some(gaussian) = self
            .as_any()
            .downcast_ref::<GaussianNumericAttributeClassObserver>()
        {
            gaussian.extra_heap_size(meter)
        } else if let Some(null_obs) = self.as_any().downcast_ref::<NullAttributeClassObserver>() {
            null_obs.extra_heap_size(meter)
        } else {
            0
        }
    }
}
