use crate::classifiers::hoeffding_tree::hoeffding_tree::HoeffdingTree;
use crate::classifiers::hoeffding_tree::nodes::FoundNode;
use crate::classifiers::hoeffding_tree::nodes::LearningNode;
use crate::classifiers::hoeffding_tree::nodes::Node;
use crate::core::instances::Instance;
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

pub struct InactiveLearningNode {
    observed_class_distribution: Vec<f64>,
}

impl InactiveLearningNode {
    pub fn new(observed_class_distribution: Vec<f64>) -> Self {
        Self {
            observed_class_distribution,
        }
    }

    pub fn num_non_zero_entries(vec: &Vec<f64>) -> usize {
        vec.iter().filter(|&&x| x != 0.0).count()
    }
}

impl Node for InactiveLearningNode {
    fn get_observed_class_distribution(&self) -> &Vec<f64> {
        &self.observed_class_distribution
    }

    fn is_leaf(&self) -> bool {
        true
    }

    fn filter_instance_to_leaf(
        &self,
        self_arc: Rc<RefCell<dyn Node>>,
        _instance: &dyn Instance,
        parent: Option<Rc<RefCell<dyn Node>>>,
        parent_branch: isize,
    ) -> FoundNode {
        FoundNode::new(Some(self_arc), parent, parent_branch)
    }

    fn get_observed_class_distribution_at_leaves_reachable_through_this_node(&self) -> Vec<f64> {
        self.observed_class_distribution.clone()
    }

    fn get_class_votes(&self, instance: &dyn Instance, hoeffding_tree: &HoeffdingTree) -> Vec<f64> {
        self.observed_class_distribution.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn observed_class_distribution_is_pure(&self) -> bool {
        Self::num_non_zero_entries(&self.observed_class_distribution) < 2
    }
    fn calc_memory_size(&self) -> usize {
        let mut total = size_of::<Self>();

        total += size_of::<Vec<f64>>();
        total += self.observed_class_distribution.len() * size_of::<f64>();

        total
    }

    fn calc_memory_size_including_subtree(&self) -> usize {
        self.calc_memory_size()
    }
}

impl LearningNode for InactiveLearningNode {
    fn learn_from_instance(&mut self, instance: &dyn Instance, hoeffding_tree: &HoeffdingTree) {
        if let Some(value) = instance.class_value() {
            let weight = instance.weight();
            if value as usize >= self.observed_class_distribution.len() {
                self.observed_class_distribution
                    .resize(value as usize + 1, 0.0);
            }
            self.observed_class_distribution[value as usize] += weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifiers::hoeffding_tree::LeafPredictionOption;
    use crate::core::attributes::Attribute;
    use crate::core::instance_header::InstanceHeader;
    use std::io::Error;

    struct MockInstance {
        values: Vec<f64>,
        class_idx: usize,
        class_val: Option<f64>,
        weight: f64,
    }

    impl MockInstance {
        fn new(values: Vec<f64>, class_idx: usize, class_val: Option<f64>, weight: f64) -> Self {
            Self {
                values,
                class_idx,
                class_val,
                weight,
            }
        }
    }

    impl Instance for MockInstance {
        fn weight(&self) -> f64 {
            self.weight
        }
        fn set_weight(&mut self, _new_value: f64) -> Result<(), Error> {
            Ok(())
        }
        fn value_at_index(&self, index: usize) -> Option<f64> {
            self.values.get(index).copied()
        }
        fn set_value_at_index(&mut self, _index: usize, _new_value: f64) -> Result<(), Error> {
            Ok(())
        }
        fn is_missing_at_index(&self, _index: usize) -> Result<bool, Error> {
            Ok(false)
        }
        fn attribute_at_index(&self, _index: usize) -> Option<&dyn Attribute> {
            None
        }
        fn index_of_attribute(&self, _attribute: &dyn Attribute) -> Option<usize> {
            None
        }
        fn number_of_attributes(&self) -> usize {
            self.values.len()
        }
        fn class_index(&self) -> usize {
            self.class_idx
        }
        fn class_value(&self) -> Option<f64> {
            self.class_val
        }
        fn set_class_value(&mut self, _new_value: f64) -> Result<(), Error> {
            Ok(())
        }
        fn is_class_missing(&self) -> bool {
            false
        }
        fn number_of_classes(&self) -> usize {
            2
        }
        fn to_vec(&self) -> Vec<f64> {
            self.values.clone()
        }
        fn header(&self) -> &InstanceHeader {
            unimplemented!()
        }
    }

    #[test]
    fn test_initialization_and_distribution() {
        let node = InactiveLearningNode::new(vec![2.0, 3.0]);
        assert_eq!(node.get_observed_class_distribution(), &vec![2.0, 3.0]);
        assert!(node.is_leaf());
    }

    #[test]
    fn test_num_non_zero_entries() {
        let vec = vec![0.0, 1.0, 2.0, 0.0, 3.0];
        assert_eq!(InactiveLearningNode::num_non_zero_entries(&vec), 3);
    }

    #[test]
    fn test_observed_class_distribution_is_pure_and_impure() {
        let pure_node = InactiveLearningNode::new(vec![4.0, 0.0]);
        assert!(pure_node.observed_class_distribution_is_pure());

        let impure_node = InactiveLearningNode::new(vec![2.0, 2.0]);
        assert!(!impure_node.observed_class_distribution_is_pure());
    }

    #[test]
    fn test_learn_from_instance_updates_distribution() {
        let mut node = InactiveLearningNode::new(vec![0.0, 0.0, 0.0]);
        let tree =
            HoeffdingTree::new_with_only_leaf_prediction(LeafPredictionOption::MajorityClass);
        let instance = MockInstance::new(vec![1.0, 2.0], 1, Some(2.0), 1.5);

        node.learn_from_instance(&instance, &tree);
        assert_eq!(node.get_observed_class_distribution()[2], 1.5);
    }

    #[test]
    fn test_learn_from_instance_with_high_class_index_expands_vector() {
        let mut node = InactiveLearningNode::new(vec![0.0, 0.0]);
        let tree =
            HoeffdingTree::new_with_only_leaf_prediction(LeafPredictionOption::MajorityClass);
        let instance = MockInstance::new(vec![0.0], 0, Some(5.0), 1.0);

        node.learn_from_instance(&instance, &tree);

        assert_eq!(node.get_observed_class_distribution().len(), 6);
        assert_eq!(node.get_observed_class_distribution()[5], 1.0);
    }

    #[test]
    fn test_get_class_votes_returns_distribution() {
        let node = InactiveLearningNode::new(vec![1.0, 2.0, 3.0]);
        let tree =
            HoeffdingTree::new_with_only_leaf_prediction(LeafPredictionOption::MajorityClass);
        let instance = MockInstance::new(vec![0.0], 0, Some(1.0), 1.0);

        let votes = node.get_class_votes(&instance, &tree);
        assert_eq!(votes, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_filter_instance_to_leaf_returns_self_wrapped() {
        let node = Rc::new(RefCell::new(InactiveLearningNode::new(vec![1.0])));
        let instance = MockInstance::new(vec![1.0], 0, Some(0.0), 1.0);
        let found = node.borrow().filter_instance_to_leaf(
            node.clone() as Rc<RefCell<dyn Node>>,
            &instance,
            None,
            0,
        );

        assert!(found.get_node().is_some());
    }

    #[test]
    fn test_calc_byte_size_non_zero() {
        let node = InactiveLearningNode::new(vec![1.0, 2.0, 3.0]);
        let byte_size = node.calc_memory_size();
        assert!(byte_size > 0);
    }

    #[test]
    fn test_get_observed_class_distribution_at_leaves_returns_clone() {
        let node = InactiveLearningNode::new(vec![1.0, 2.0]);
        let clone = node.get_observed_class_distribution_at_leaves_reachable_through_this_node();
        assert_eq!(clone, vec![1.0, 2.0]);
    }
}
