use crate::core::attributes::Attribute;
use crate::utils::memory::{MemoryMeter, MemorySized};
use std::any::Any;
use std::mem::size_of;

#[derive(Clone)]
pub struct NumericAttribute {
    pub name: String,
    pub values: Vec<u32>,
}

impl NumericAttribute {
    pub fn new(name: String) -> NumericAttribute {
        NumericAttribute {
            name,
            values: Vec::new(),
        }
    }

    pub fn with_values(name: String, values: Vec<u32>) -> NumericAttribute {
        NumericAttribute { name, values }
    }
}

impl Attribute for NumericAttribute {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn arff_representation(&self) -> String {
        let numeric = self.as_any().downcast_ref::<NumericAttribute>().unwrap();
        format!("@attribute {} numeric", numeric.name())
    }

    fn calc_memory_size(&self) -> usize {
        MemoryMeter::measure_root(self)
    }
}

impl MemorySized for NumericAttribute {
    fn inline_size(&self) -> usize {
        size_of::<Self>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        let mut total = 0;
        total += meter.measure_field(&self.name);
        total += meter.measure_field(&self.values);
        total
    }
}
