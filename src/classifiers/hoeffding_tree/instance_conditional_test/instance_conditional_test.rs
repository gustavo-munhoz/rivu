use crate::classifiers::hoeffding_tree::instance_conditional_test::nominal_attribute_binary_test::NominalAttributeBinaryTest;
use crate::classifiers::hoeffding_tree::instance_conditional_test::nominal_attribute_multiway_test::NominalAttributeMultiwayTest;
use crate::classifiers::hoeffding_tree::instance_conditional_test::numeric_attribute_binary_test::NumericAttributeBinaryTest;
use crate::core::instances::Instance;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;

pub trait InstanceConditionalTest: Any {
    fn branch_for_instance(&self, instance: &dyn Instance) -> Option<usize>;
    fn result_known_for_instance(&self, instance: &dyn Instance) -> bool;
    fn max_branches(&self) -> usize;
    fn get_atts_test_depends_on(&self) -> Vec<usize>;
    fn calc_memory_size(&self) -> usize;
    fn clone_box(&self) -> Box<dyn InstanceConditionalTest>;
    fn as_any(&self) -> &dyn Any;
}

impl Clone for Box<dyn InstanceConditionalTest> {
    fn clone(&self) -> Box<dyn InstanceConditionalTest> {
        self.clone_box()
    }
}

impl MemorySized for dyn InstanceConditionalTest {
    fn inline_size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        if let Some(nom_bin) = self.as_any().downcast_ref::<NominalAttributeBinaryTest>() {
            nom_bin.extra_heap_size(meter)
        } else if let Some(num_bin) = self.as_any().downcast_ref::<NumericAttributeBinaryTest>() {
            num_bin.extra_heap_size(meter)
        } else if let Some(nom_multi) = self.as_any().downcast_ref::<NominalAttributeMultiwayTest>()
        {
            nom_multi.extra_heap_size(meter)
        } else {
            0
        }
    }
}
