use crate::classifiers::Classifier;
use crate::core::attributes::NominalAttribute;
use crate::core::instance_header::InstanceHeader;
use crate::core::instances::Instance;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub struct TrainSpyHandle(Arc<AtomicU64>);
impl TrainSpyHandle {
    pub fn count(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

pub struct TrainSpyClassifier {
    count: Arc<AtomicU64>,
    num_classes: usize,
}

impl TrainSpyClassifier {
    pub fn new() -> (Self, TrainSpyHandle) {
        let counter = Arc::new(AtomicU64::new(0));
        (
            Self {
                count: counter.clone(),
                num_classes: 2,
            },
            TrainSpyHandle(counter),
        )
    }
}

impl Classifier for TrainSpyClassifier {
    fn get_votes_for_instance(&self, inst: &dyn Instance) -> Vec<f64> {
        let y = inst.class_value().unwrap_or_default() as usize;
        let mut v = vec![0.0; self.num_classes.max(2)];
        if y < v.len() {
            v[y] = 1.0;
        }
        v
    }

    fn set_model_context(&mut self, h: Arc<InstanceHeader>) {
        self.num_classes = h
            .attribute_at_index(h.class_index())
            .and_then(|a| a.as_any().downcast_ref::<NominalAttribute>())
            .map(|n| n.values.len())
            .unwrap_or(2);
    }

    fn train_on_instance(&mut self, _inst: &dyn Instance) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    fn calc_memory_size(&self) -> usize {
        unimplemented!()
    }
}
