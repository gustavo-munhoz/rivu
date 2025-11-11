use crate::classifiers::hoeffding_tree::instance_conditional_test::instance_conditional_test::InstanceConditionalTest;
use crate::core::instances::Instance;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::mem::size_of;

#[derive(Clone)]
pub struct NumericAttributeBinaryTest {
    attribute_index: usize,
    attribute_value: f64,
    equals_passes_test: bool,
}

impl NumericAttributeBinaryTest {
    pub fn new(attribute_index: usize, attribute_value: f64, equals_passes_test: bool) -> Self {
        Self {
            attribute_index,
            attribute_value,
            equals_passes_test,
        }
    }
}

impl InstanceConditionalTest for NumericAttributeBinaryTest {
    fn branch_for_instance(&self, instance: &dyn Instance) -> Option<usize> {
        let index = self.attribute_index;

        if instance.is_missing_at_index(index).unwrap_or(true) {
            return None;
        }

        let value = instance.value_at_index(self.attribute_index)?;

        if value == self.attribute_value {
            return Some(if self.equals_passes_test { 0 } else { 1 });
        }
        if value < self.attribute_value {
            return Some(0);
        }
        Some(1)
    }

    fn result_known_for_instance(&self, instance: &dyn Instance) -> bool {
        self.branch_for_instance(instance).is_some_and(|b| b == 0)
    }

    fn max_branches(&self) -> usize {
        2
    }

    fn get_atts_test_depends_on(&self) -> Vec<usize> {
        vec![self.attribute_index]
    }

    fn calc_memory_size(&self) -> usize {
        MemoryMeter::measure_root(self)
    }

    fn clone_box(&self) -> Box<dyn InstanceConditionalTest> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MemorySized for NumericAttributeBinaryTest {
    fn inline_size(&self) -> usize {
        size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            unimplemented!()
        }
        fn value_at_index(&self, index: usize) -> Option<f64> {
            self.values.get(index).copied()
        }
        fn set_value_at_index(&mut self, _index: usize, _new_value: f64) -> Result<(), Error> {
            unimplemented!()
        }
        fn is_missing_at_index(&self, index: usize) -> Result<bool, Error> {
            if index < self.values.len() {
                Ok(self.values[index].is_nan())
            } else {
                Err(Error::new(std::io::ErrorKind::InvalidInput, "oob"))
            }
        }
        fn attribute_at_index(&self, _index: usize) -> Option<&dyn Attribute> {
            unimplemented!()
        }
        fn index_of_attribute(&self, _attribute: &dyn Attribute) -> Option<usize> {
            unimplemented!()
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
            unimplemented!()
        }
        fn is_class_missing(&self) -> bool {
            unimplemented!()
        }
        fn number_of_classes(&self) -> usize {
            unimplemented!()
        }
        fn to_vec(&self) -> Vec<f64> {
            unimplemented!()
        }
        fn header(&self) -> &InstanceHeader {
            unimplemented!()
        }
    }

    #[test]
    fn test_branch_for_instance_returns_zero_when_value_less_then_split() {
        let test = NumericAttributeBinaryTest::new(0, 5.0, true);
        let instance = MockInstance::new(vec![3.0], 1, None, 1.0);

        let branch = test.branch_for_instance(&instance).unwrap();
        assert_eq!(branch, 0);
    }

    #[test]
    fn test_branch_for_instance_returns_one_when_value_greater_then_split() {
        let test = NumericAttributeBinaryTest::new(0, 5.0, true);
        let instance = MockInstance::new(vec![8.0], 1, None, 1.0);

        let branch = test.branch_for_instance(&instance).unwrap();
        assert_eq!(branch, 1);
    }

    #[test]
    fn test_branch_for_instance_returns_zero_when_value_equals_split_and_equals_passes_true() {
        let test = NumericAttributeBinaryTest::new(0, 5.0, true);
        let instance = MockInstance::new(vec![5.0], 1, Some(0.0), 1.0);

        let branch = test.branch_for_instance(&instance).unwrap();
        assert_eq!(branch, 0);
    }

    #[test]
    fn test_branch_for_instance_returns_one_when_value_equals_split_and_equals_passes_false() {
        let test = NumericAttributeBinaryTest::new(0, 5.0, false);
        let instance = MockInstance::new(vec![5.0], 1, Some(0.0), 1.0);

        let branch = test.branch_for_instance(&instance).unwrap();
        assert_eq!(branch, 1);
    }

    #[test]
    fn test_branch_for_instance_returns_none_when_value_missing() {
        let test = NumericAttributeBinaryTest::new(1, 5.0, true);
        let instance = MockInstance::new(vec![0.0], 1, Some(0.0), 1.0);

        let branch = test.branch_for_instance(&instance);
        assert!(branch.is_none());
    }

    #[test]
    fn test_result_known_for_instance_true_only_if_branch_zero() {
        let test = NumericAttributeBinaryTest::new(0, 5.0, true);
        let instance_low = MockInstance::new(vec![3.0], 1, Some(0.0), 1.0);
        let instance_equal = MockInstance::new(vec![5.0], 1, Some(0.0), 1.0);
        let instance_high = MockInstance::new(vec![10.0], 1, Some(0.0), 1.0);

        assert!(test.result_known_for_instance(&instance_low));
        assert!(test.result_known_for_instance(&instance_equal));
        assert!(!test.result_known_for_instance(&instance_high));
    }

    #[test]
    fn test_max_branches_is_two() {
        let test = NumericAttributeBinaryTest::new(0, 3.5, true);
        assert_eq!(test.max_branches(), 2);
    }

    #[test]
    fn test_get_atts_test_depends_on_returns_correct_attribute() {
        let test = NumericAttributeBinaryTest::new(4, 2.3, true);
        let atts = test.get_atts_test_depends_on();
        assert_eq!(atts, vec![4]);
    }

    #[test]
    fn test_calc_byte_size_returns_correct_size() {
        let test = NumericAttributeBinaryTest::new(0, 3.5, true);
        assert_eq!(
            test.calc_memory_size(),
            size_of::<NumericAttributeBinaryTest>()
        );
    }

    #[test]
    fn test_clone_box_returns_correct_clone() {
        let test = NumericAttributeBinaryTest::new(0, 5.0, true);
        let clone = test.clone_box();

        let instance = MockInstance::new(vec![4.0], 1, Some(0.0), 1.0);
        assert_eq!(
            test.branch_for_instance(&instance),
            clone.branch_for_instance(&instance)
        );
    }
}
