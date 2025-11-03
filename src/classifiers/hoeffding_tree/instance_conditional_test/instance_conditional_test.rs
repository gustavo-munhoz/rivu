use crate::core::instances::Instance;

pub trait InstanceConditionalTest {
    fn branch_for_instance(&self, instance: &dyn Instance) -> Option<usize>;
    fn result_known_for_instance(&self, instance: &dyn Instance) -> bool;
    fn max_branches(&self) -> usize;
    fn get_atts_test_depends_on(&self) -> Vec<usize>;
    fn calc_memory_size(&self) -> usize;
    fn clone_box(&self) -> Box<dyn InstanceConditionalTest>;
}

impl Clone for Box<dyn InstanceConditionalTest> {
    fn clone(&self) -> Box<dyn InstanceConditionalTest> {
        self.clone_box()
    }
}
