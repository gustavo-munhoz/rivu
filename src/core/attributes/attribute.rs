use super::{NominalAttribute, NumericAttribute};
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::sync::Arc;

pub type AttributeRef = Arc<dyn Attribute + Send + Sync>;

pub trait Attribute: Any + Send + Sync {
    fn name(&self) -> String;

    fn as_any(&self) -> &dyn Any;

    fn arff_representation(&self) -> String;
    fn calc_memory_size(&self) -> usize;
}

impl MemorySized for dyn Attribute + Send + Sync {
    fn inline_size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        if let Some(nominal) = self.as_any().downcast_ref::<NominalAttribute>() {
            nominal.extra_heap_size(meter)
        } else if let Some(numeric) = self.as_any().downcast_ref::<NumericAttribute>() {
            numeric.extra_heap_size(meter)
        } else {
            0
        }
    }
}
