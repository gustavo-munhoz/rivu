use crate::core::attributes::Attribute;
use std::any::Any;

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
        let mut total: usize = 0;

        total += size_of::<Self>();

        total += self.name.capacity();

        total += size_of::<Vec<u32>>();

        total += self.values.capacity() * size_of::<u32>();

        total
    }
}
