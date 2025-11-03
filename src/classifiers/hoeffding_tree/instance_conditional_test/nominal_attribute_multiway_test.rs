use crate::classifiers::hoeffding_tree::instance_conditional_test::instance_conditional_test::InstanceConditionalTest;
use crate::core::instances::Instance;

#[derive(Clone)]
pub struct NominalAttributeMultiwayTest {
    attribute_index: usize,
}

impl NominalAttributeMultiwayTest {
    pub fn new(attribute_index: usize) -> Self {
        Self { attribute_index }
    }
}

impl InstanceConditionalTest for NominalAttributeMultiwayTest {
    fn branch_for_instance(&self, instance: &dyn Instance) -> Option<usize> {
        if instance
            .is_missing_at_index(self.attribute_index)
            .unwrap_or(true)
        {
            return None;
        }

        Some(instance.value_at_index(self.attribute_index)? as usize)
    }

    fn result_known_for_instance(&self, instance: &dyn Instance) -> bool {
        self.branch_for_instance(instance).is_some_and(|b| b == 0)
    }

    fn max_branches(&self) -> usize {
        usize::MAX
    }

    fn get_atts_test_depends_on(&self) -> Vec<usize> {
        vec![self.attribute_index]
    }

    fn calc_memory_size(&self) -> usize {
        size_of::<Self>()
    }

    fn clone_box(&self) -> Box<dyn InstanceConditionalTest> {
        Box::new(self.clone())
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
    fn test_branch_for_instance_returns_value_as_branch_index() {
        let test = NominalAttributeMultiwayTest::new(1);
        let instance = MockInstance::new(vec![0.0, 3.0, 2.0], 2, Some(0.0), 1.0);

        let branch = test.branch_for_instance(&instance);
        assert!(branch.is_some());
    }

    #[test]
    fn test_branch_for_instance_returns_none_for_missing() {
        let test = NominalAttributeMultiwayTest::new(2);
        let instance = MockInstance::new(vec![0.0, 2.0], 2, Some(0.0), 1.0);

        let branch = test.branch_for_instance(&instance);
        assert!(branch.is_none());
    }

    #[test]
    fn test_result_known_for_instance_true_only_if_branch_zero() {
        let test = NominalAttributeMultiwayTest::new(0);

        let instance_zero = MockInstance::new(vec![0.0, 2.0], 1, Some(0.0), 1.0);
        let instance_nonzero = MockInstance::new(vec![1.0, 2.0], 1, Some(0.0), 1.0);
        let instance_missing = MockInstance::new(vec![], 1, Some(0.0), 1.0);

        assert!(test.result_known_for_instance(&instance_zero));
        assert!(!test.result_known_for_instance(&instance_nonzero));
        assert!(!test.result_known_for_instance(&instance_missing));
    }

    #[test]
    fn test_max_branches_is_usize_max() {
        let test = NominalAttributeMultiwayTest::new(0);
        assert_eq!(test.max_branches(), usize::MAX);
    }

    #[test]
    fn test_get_atts_test_depends_on_returns_correct_attribute() {
        let test = NominalAttributeMultiwayTest::new(7);
        assert_eq!(test.get_atts_test_depends_on(), vec![7]);
    }

    fn test_calc_byte_size() {
        let test = NominalAttributeMultiwayTest::new(0);
        assert_eq!(
            test.calc_memory_size(),
            std::mem::size_of::<NominalAttributeMultiwayTest>()
        );
    }

    fn test_clone_box() {
        let test = NominalAttributeMultiwayTest::new(1);
        let cloned_test = test.clone_box();

        let instance = MockInstance::new(vec![0.0, 5.0], 2, Some(0.0), 1.0);
        assert_eq!(
            test.branch_for_instance(&instance),
            cloned_test.branch_for_instance(&instance)
        );
    }
}
