use std::any::Any;
use std::sync::Arc;

pub type AttributeRef = Arc<dyn Attribute + Send + Sync>;

pub trait Attribute: Any + Send + Sync {
    fn name(&self) -> String;

    fn as_any(&self) -> &dyn Any;

    fn arff_representation(&self) -> String;
    fn calc_memory_size(&self) -> usize;
}
