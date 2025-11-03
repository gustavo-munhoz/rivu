use crate::classifiers::Classifier;
use crate::core::attributes::NominalAttribute;
use crate::core::instance_header::InstanceHeader;
use crate::core::instances::Instance;
use std::sync::Arc;

#[derive(Default)]
pub struct OracleClassifier {
    num_classes: usize,
}

impl Classifier for OracleClassifier {
    fn get_votes_for_instance(&self, instance: &dyn Instance) -> Vec<f64> {
        let y = instance.class_value().unwrap_or_default() as usize;
        let mut v = vec![0.0; self.num_classes.max(2)];
        if y < v.len() {
            v[y] = 1.0;
        }
        v
    }

    fn set_model_context(&mut self, header: Arc<InstanceHeader>) {
        self.num_classes = header
            .attribute_at_index(header.class_index())
            .and_then(|a| a.as_any().downcast_ref::<NominalAttribute>())
            .map(|n| n.values.len())
            .unwrap_or(2)
    }

    fn train_on_instance(&mut self, instance: &dyn Instance) {}

    fn calc_memory_size(&self) -> usize {
        size_of::<Self>()
    }
}
