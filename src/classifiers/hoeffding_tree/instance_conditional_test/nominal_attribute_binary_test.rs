use crate::classifiers::hoeffding_tree::instance_conditional_test::instance_conditional_test::InstanceConditionalTest;
use crate::core::instances::Instance;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::mem::size_of;

#[derive(Clone)]
pub struct NominalAttributeBinaryTest {
    attribute_index: usize,
    attribute_value: usize,
}

impl NominalAttributeBinaryTest {
    pub fn new(attribute_index: usize, attribute_value: usize) -> Self {
        Self {
            attribute_index,
            attribute_value,
        }
    }
}

impl InstanceConditionalTest for NominalAttributeBinaryTest {
    fn branch_for_instance(&self, instance: &dyn Instance) -> Option<usize> {
        let index = if self.attribute_index < instance.class_index() {
            self.attribute_index
        } else {
            self.attribute_index + 1
        };

        if instance.is_missing_at_index(index).unwrap_or(true) {
            return None;
        }

        let value = instance.value_at_index(index)?;

        if value as usize == self.attribute_value {
            Some(0)
        } else {
            Some(1)
        }
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

impl MemorySized for NominalAttributeBinaryTest {
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
    fn test_branch_for_instance_returns_zero_when_value_matches() {
        let test = NominalAttributeBinaryTest::new(0, 1);
        let instance = MockInstance::new(vec![1.0, 0.0, 0.0], 2, Some(0.0), 1.0);

        let branch = test.branch_for_instance(&instance).unwrap();
        assert_eq!(branch, 0);
    }

    #[test]
    fn test_branch_for_instance_returns_one_when_value_differs() {
        let test = NominalAttributeBinaryTest::new(0, 2);
        let instance = MockInstance::new(vec![1.0, 0.0], 2, Some(0.0), 1.0);

        let branch = test.branch_for_instance(&instance).unwrap();
        assert_eq!(branch, 1);
    }

    #[test]
    fn test_branch_for_instance_returns_none_when_missing() {
        let test = NominalAttributeBinaryTest::new(2, 1);
        let instance = MockInstance::new(vec![0.0, 1.0], 3, Some(0.0), 1.0);

        let branch = test.branch_for_instance(&instance);
        assert!(branch.is_none());
    }

    #[test]
    fn test_branch_for_instance_with_class_index_shift() {
        let test = NominalAttributeBinaryTest::new(2, 1);
        let instance = MockInstance::new(vec![0.0, 1.0, 1.0, 1.0], 1, Some(0.0), 1.0);

        let branch = test.branch_for_instance(&instance).unwrap();
        assert_eq!(branch, 0);
    }

    #[test]
    fn test_result_known_for_instance_true_only_if_branch_zero() {
        let test = NominalAttributeBinaryTest::new(1, 1);
        let instance_match = MockInstance::new(vec![1.0, 1.0], 2, Some(0.0), 1.0);
        let instance_diff = MockInstance::new(vec![2.0, 2.0], 2, Some(0.0), 1.0);
        let instance_missing = MockInstance::new(vec![0.0], 2, Some(0.0), 1.0);

        assert!(test.result_known_for_instance(&instance_match));
        assert!(!test.result_known_for_instance(&instance_diff));
        assert!(!test.result_known_for_instance(&instance_missing));
    }

    #[test]
    fn test_max_branches() {
        let test = NominalAttributeBinaryTest::new(0, 1);
        assert_eq!(test.max_branches(), 2);
    }

    #[test]
    fn test_get_atts_test_depends_on_return_correct_attribute() {
        let test = NominalAttributeBinaryTest::new(5, 3);
        assert_eq!(test.get_atts_test_depends_on(), vec![5]);
    }

    #[test]
    fn test_calc_byte_size() {
        let test = NominalAttributeBinaryTest::new(0, 1);
        assert_eq!(
            test.calc_memory_size(),
            std::mem::size_of::<NominalAttributeBinaryTest>()
        );
    }

    #[test]
    fn test_clone_box() {
        let test = NominalAttributeBinaryTest::new(0, 1);
        let clone = test.clone_box();

        let instance = MockInstance::new(vec![1.0, 0.0], 2, Some(0.0), 1.0);
        assert_eq!(
            test.branch_for_instance(&instance),
            clone.branch_for_instance(&instance)
        );
    }
}
