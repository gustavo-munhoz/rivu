use crate::core::attributes::{Attribute, AttributeRef, NominalAttribute};
use std::fmt;

pub struct InstanceHeader {
    relation_name: String,
    pub attributes: Vec<AttributeRef>,
    class_index: usize,
}

impl InstanceHeader {
    pub fn new(
        relation_name: String,
        attributes: Vec<AttributeRef>,
        class_index: usize,
    ) -> InstanceHeader {
        InstanceHeader {
            relation_name,
            attributes,
            class_index,
        }
    }

    pub fn class_attribute(&self, index: usize) -> &dyn Attribute {
        self.attributes[index].as_ref()
    }

    pub fn number_of_attributes(&self) -> usize {
        self.attributes.len()
    }

    pub fn relation_name(&self) -> &str {
        &self.relation_name
    }

    pub fn attribute_at_index(&self, index: usize) -> Option<&dyn Attribute> {
        if index < self.attributes.len() {
            Some(self.attributes[index].as_ref())
        } else {
            None
        }
    }

    pub fn index_of_attribute(&self, name: &str) -> Option<usize> {
        for (i, attr) in self.attributes.iter().enumerate() {
            if attr.name() == name {
                return Some(i);
            }
        }
        None
    }

    pub fn class_index(&self) -> usize {
        self.class_index
    }

    pub fn number_of_classes(&self) -> usize {
        if self.class_index < self.attributes.len() {
            if let Some(nominal_attr) = self.attributes[self.class_index]
                .as_any()
                .downcast_ref::<NominalAttribute>()
            {
                return nominal_attr.values.len();
            }
        }
        0
    }

    pub fn calc_memory_size(&self) -> usize {
        let mut total: usize = 0;

        total += size_of::<Self>();

        total += self.relation_name.capacity();

        total += self.attributes.capacity() * size_of::<AttributeRef>();

        for attr_arc in &self.attributes {
            total += attr_arc.calc_memory_size();
        }

        total
    }
}

impl fmt::Debug for InstanceHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstanceHeader")
            .field("relation_name", &self.relation_name)
            .field("class_index", &self.class_index)
            .field("n_attributes", &self.attributes.len())
            .finish()
    }
}
