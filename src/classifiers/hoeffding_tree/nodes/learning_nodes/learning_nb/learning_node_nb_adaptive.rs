use crate::classifiers::NaiveBayes;
use crate::classifiers::attribute_class_observers::AttributeClassObserver;
use crate::classifiers::conditional_tests::attribute_split_suggestion::AttributeSplitSuggestion;
use crate::classifiers::hoeffding_tree::hoeffding_tree::HoeffdingTree;
use crate::classifiers::hoeffding_tree::nodes::LearningNode;
use crate::classifiers::hoeffding_tree::nodes::Node;
use crate::classifiers::hoeffding_tree::nodes::found_node::FoundNode;
use crate::classifiers::hoeffding_tree::split_criteria::SplitCriterion;
use crate::core::attributes::NominalAttribute;
use crate::core::instances::Instance;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::cell::RefCell;
use std::mem::size_of;
use std::rc::Rc;

pub struct LearningNodeNBAdaptive {
    observed_class_distribution: Vec<f64>,
    weight_seen_at_last_split_evaluation: f64,
    attribute_observers: Vec<Option<Box<dyn AttributeClassObserver>>>,
    is_initialized: bool,
    mc_correct_weight: f64,
    nb_correct_weight: f64,
}

impl LearningNodeNBAdaptive {
    pub fn new(observed_class_distribution: Vec<f64>) -> Self {
        let weight_seen = observed_class_distribution.iter().sum();
        Self {
            observed_class_distribution,
            weight_seen_at_last_split_evaluation: weight_seen,
            attribute_observers: Vec::new(),
            is_initialized: false,
            mc_correct_weight: 0.0,
            nb_correct_weight: 0.0,
        }
    }

    pub fn get_weight_seen(&self) -> f64 {
        self.observed_class_distribution.iter().sum()
    }

    pub fn get_weight_seen_at_last_split_evaluation(&self) -> f64 {
        self.weight_seen_at_last_split_evaluation
    }

    pub fn set_weight_seen_at_last_split_evaluation(&mut self, weight: f64) {
        self.weight_seen_at_last_split_evaluation = weight;
    }

    fn max_index(dist: &[f64]) -> Option<usize> {
        if dist.is_empty() {
            return None;
        }

        let mut max_i = 0usize;
        let mut max_v = dist[0];
        for (i, &v) in dist.iter().enumerate().skip(1) {
            if v > max_v {
                max_v = v;
                max_i = i;
            }
        }
        Some(max_i)
    }

    fn super_learn_from_instance(
        &mut self,
        instance: &dyn Instance,
        hoeffding_tree: &HoeffdingTree,
    ) {
        if !self.is_initialized {
            self.attribute_observers = (0..instance.number_of_attributes()).map(|_| None).collect();
            self.is_initialized = true;
        }

        if let Some(class_index) = instance.class_value() {
            let weight = instance.weight();
            let idx = class_index as usize;
            if idx >= self.observed_class_distribution.len() {
                self.observed_class_distribution.resize(idx + 1, 0.0);
            }
            self.observed_class_distribution[idx] += weight;
        }

        for i in 0..instance.number_of_attributes() - 1 {
            let instance_attribute_index =
                HoeffdingTree::model_attribute_index_to_instance_attribute_index(i, instance);

            if self.attribute_observers[i].is_none() {
                if let Some(attribute) = instance.attribute_at_index(instance_attribute_index) {
                    let observer: Box<dyn AttributeClassObserver> =
                        if attribute.as_any().is::<NominalAttribute>() {
                            hoeffding_tree.new_nominal_class_observer()
                        } else {
                            hoeffding_tree.new_numeric_class_observer()
                        };
                    self.attribute_observers[i] = Some(observer);
                }
            }

            if let Some(observer) = self.attribute_observers[i].as_mut() {
                if let (Some(class_index), Some(value)) = (
                    instance.class_value(),
                    instance.value_at_index(instance_attribute_index),
                ) {
                    observer.observe_attribute_class(
                        value,
                        class_index as usize,
                        instance.weight(),
                    );
                }
            }
        }
    }

    pub fn num_non_zero_entries(vec: &Vec<f64>) -> usize {
        vec.iter().filter(|&&x| x != 0.0).count()
    }

    pub fn get_best_split_suggestions(
        &self,
        criterion: &dyn SplitCriterion,
        ht: &HoeffdingTree,
    ) -> Vec<AttributeSplitSuggestion> {
        let mut best_suggestions: Vec<AttributeSplitSuggestion> = Vec::new();
        let pre_split_distribution = self.observed_class_distribution.clone();
        if !ht.get_no_pre_prune_option() {
            let merit = criterion
                .get_merit_of_split(&pre_split_distribution, &[pre_split_distribution.clone()]);
            best_suggestions.push(AttributeSplitSuggestion::new(
                None,
                vec![pre_split_distribution.clone()],
                merit,
            ));
        }

        for (i, obs_opt) in self.attribute_observers.iter().enumerate() {
            if let Some(obs) = obs_opt {
                if let Some(best_suggestion) = obs.get_best_evaluated_split_suggestion(
                    criterion,
                    &pre_split_distribution,
                    i,
                    ht.get_binary_splits_option(),
                ) {
                    best_suggestions.push(best_suggestion)
                }
            }
        }
        best_suggestions
    }
}

impl Node for LearningNodeNBAdaptive {
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

    fn get_class_votes(
        &self,
        instance: &dyn Instance,
        _hoeffding_tree: &HoeffdingTree,
    ) -> Vec<f64> {
        if self.mc_correct_weight > self.nb_correct_weight {
            return self.observed_class_distribution.clone();
        }
        NaiveBayes::do_naive_bayes_prediction(
            instance,
            &self.observed_class_distribution,
            &self.attribute_observers,
        )
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
        MemoryMeter::measure_root(self)
    }

    fn calc_memory_size_including_subtree(&self) -> usize {
        self.calc_memory_size()
    }
}

impl MemorySized for LearningNodeNBAdaptive {
    fn inline_size(&self) -> usize {
        size_of::<Self>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        let mut total = 0;
        total += meter.measure_field(&self.observed_class_distribution);
        total += meter.measure_field(&self.attribute_observers);
        total
    }
}

impl LearningNode for LearningNodeNBAdaptive {
    fn learn_from_instance(&mut self, instance: &dyn Instance, hoeffding_tree: &HoeffdingTree) {
        if let Some(true_class) = instance.class_value() {
            let weight = instance.weight();

            if let Some(predicted_mc) = Self::max_index(&self.observed_class_distribution) {
                if predicted_mc == true_class as usize {
                    self.mc_correct_weight += weight;
                }
            }

            let nb_prediction = NaiveBayes::do_naive_bayes_prediction(
                instance,
                &self.observed_class_distribution,
                &self.attribute_observers,
            );

            if let Some(predicted_nb) = Self::max_index(&nb_prediction) {
                if predicted_nb == true_class as usize {
                    self.nb_correct_weight += weight;
                }
            }
        }

        self.super_learn_from_instance(instance, hoeffding_tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifiers::HoeffdingTree;
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

    struct MockSplitCriterion;

    impl SplitCriterion for MockSplitCriterion {
        fn get_merit_of_split(&self, _pre: &[f64], _post: &[Vec<f64>]) -> f64 {
            42.0
        }
        fn get_range_of_merit(&self, _pre: &Vec<f64>) -> f64 {
            1.0
        }

        fn as_any(&self) -> &dyn Any {
            unimplemented!()
        }
    }

    #[test]
    fn test_initialization_and_weight_sum() {
        let node = LearningNodeNBAdaptive::new(vec![2.0, 3.0]);
        assert_eq!(node.get_weight_seen(), 5.0);
        assert_eq!(node.get_weight_seen_at_last_split_evaluation(), 5.0);
    }

    #[test]
    fn test_num_non_zero_entries() {
        let v = vec![0.0, 1.0, 2.0, 0.0];
        assert_eq!(LearningNodeNBAdaptive::num_non_zero_entries(&v), 2);
    }

    #[test]
    fn test_max_index() {
        assert!(LearningNodeNBAdaptive::max_index(&vec![0.1, 2.5, 1.0]) == Some(1));
        assert_eq!(LearningNodeNBAdaptive::max_index(&[]), None);
    }

    #[test]
    fn test_learn_from_instance_expands_distribution() {
        let mut node = LearningNodeNBAdaptive::new(vec![0.0]);
        let tree =
            HoeffdingTree::new_with_only_leaf_prediction(LeafPredictionOption::AdaptiveNaiveBayes);
        let instance = MockInstance::new(vec![1.0, 2.0], 1, Some(3.0), 2.0);
        node.learn_from_instance(&instance, &tree);

        assert_eq!(node.get_observed_class_distribution().len(), 4);
        assert_eq!(node.get_observed_class_distribution()[3], 2.0);
    }

    #[test]
    fn test_get_class_votes_uses_mc_or_nb() {
        let mut node = LearningNodeNBAdaptive::new(vec![5.0, 2.0]);
        let instance = MockInstance::new(vec![0.0, 1.0], 1, Some(1.0), 1.0);
        let tree =
            HoeffdingTree::new_with_only_leaf_prediction(LeafPredictionOption::AdaptiveNaiveBayes);

        node.mc_correct_weight = 10.0;
        node.nb_correct_weight = 5.0;
        let votes_mc = node.get_class_votes(&instance, &tree);
        assert_eq!(votes_mc, vec![5.0, 2.0]);

        node.mc_correct_weight = 2.0;
        node.nb_correct_weight = 10.0;
        let votes_nb = node.get_class_votes(&instance, &tree);
        assert_eq!(votes_nb.len(), 2);
    }

    #[test]
    fn test_learn_from_instance_updates_correct_weights() {
        let mut node = LearningNodeNBAdaptive::new(vec![1.0, 5.0]);
        let tree =
            HoeffdingTree::new_with_only_leaf_prediction(LeafPredictionOption::AdaptiveNaiveBayes);
        let instance = MockInstance::new(vec![0.0, 1.0], 1, Some(1.0), 1.0);

        node.learn_from_instance(&instance, &tree);

        assert!(node.mc_correct_weight > 0.0);
        assert!(node.nb_correct_weight >= 0.0);
    }

    #[test]
    fn test_get_best_split_suggestions_returns_nonempty() {
        let node = LearningNodeNBAdaptive::new(vec![1.0, 1.0]);
        let tree =
            HoeffdingTree::new_with_only_leaf_prediction(LeafPredictionOption::AdaptiveNaiveBayes);
        let criterion = MockSplitCriterion;

        let suggestions = node.get_best_split_suggestions(&criterion, &tree);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].get_merit(), 42.0);
    }

    #[test]
    fn test_calc_byte_size_nonzero() {
        let node = LearningNodeNBAdaptive::new(vec![1.0, 2.0, 3.0]);
        let size = node.calc_memory_size();
        assert!(size > 0);
    }

    #[test]
    fn test_filter_instance_to_leaf_returns_self() {
        let node = Rc::new(RefCell::new(LearningNodeNBAdaptive::new(vec![1.0])));
        let instance = MockInstance::new(vec![1.0], 0, Some(0.0), 1.0);
        let found = node.borrow().filter_instance_to_leaf(
            node.clone() as Rc<RefCell<dyn Node>>,
            &instance,
            None,
            0,
        );
        assert!(found.get_node().is_some());
    }
}
