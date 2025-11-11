use crate::classifiers::hoeffding_tree::hoeffding_tree::HoeffdingTree;
use crate::classifiers::hoeffding_tree::nodes::found_node::FoundNode;
use crate::classifiers::hoeffding_tree::nodes::{
    ActiveLearningNode, InactiveLearningNode, LearningNodeNB, LearningNodeNBAdaptive, SplitNode,
};
use crate::core::instances::Instance;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

pub trait Node: Any {
    fn get_observed_class_distribution(&self) -> &Vec<f64>;
    fn is_leaf(&self) -> bool;
    fn filter_instance_to_leaf(
        &self,
        self_arc: Rc<RefCell<dyn Node>>,
        instance: &dyn Instance,
        parent: Option<Rc<RefCell<dyn Node>>>,
        parent_branch: isize,
    ) -> FoundNode;
    fn get_observed_class_distribution_at_leaves_reachable_through_this_node(&self) -> Vec<f64>;
    fn get_class_votes(&self, instance: &dyn Instance, hoeffding_tree: &HoeffdingTree) -> Vec<f64>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn observed_class_distribution_is_pure(&self) -> bool;
    fn calc_memory_size(&self) -> usize;
    fn calc_memory_size_including_subtree(&self) -> usize;
}

impl MemorySized for dyn Node {
    fn inline_size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        if let Some(active) = self.as_any().downcast_ref::<ActiveLearningNode>() {
            active.extra_heap_size(meter)
        } else if let Some(nb) = self.as_any().downcast_ref::<LearningNodeNB>() {
            nb.extra_heap_size(meter)
        } else if let Some(nb_adapt) = self.as_any().downcast_ref::<LearningNodeNBAdaptive>() {
            nb_adapt.extra_heap_size(meter)
        } else if let Some(inactive) = self.as_any().downcast_ref::<InactiveLearningNode>() {
            inactive.extra_heap_size(meter)
        } else if let Some(split) = self.as_any().downcast_ref::<SplitNode>() {
            split.extra_heap_size(meter)
        } else {
            0
        }
    }
}
