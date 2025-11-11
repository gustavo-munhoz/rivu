use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ptr;
use std::rc::Rc;
use std::sync::Arc;

/// Utility for estimating the memory consumed by a structure and all of its
/// reachable data.
#[derive(Default)]
pub struct MemoryMeter {
    visited: HashSet<usize>,
}

impl MemoryMeter {
    #[inline]
    pub fn new() -> Self {
        Self {
            visited: HashSet::new(),
        }
    }

    #[inline]
    fn mark<T: ?Sized>(&mut self, value: &T) -> bool {
        let ptr = ptr::from_ref(value) as *const () as usize;
        self.visited.insert(ptr)
    }

    #[inline]
    fn measure<T: MemorySized + ?Sized>(&mut self, value: &T) -> usize {
        if self.mark(value) {
            value.inline_size() + value.extra_heap_size(self)
        } else {
            0
        }
    }

    #[inline]
    pub fn measure_root<T: MemorySized + ?Sized>(value: &T) -> usize {
        let mut meter = MemoryMeter::new();
        meter.measure(value)
    }

    #[inline]
    pub fn measure_field<T: MemorySized + ?Sized>(&mut self, value: &T) -> usize {
        let total = self.measure(value);
        total.saturating_sub(value.inline_size())
    }

    #[inline]
    pub unsafe fn measure_shared<T: MemorySized + ?Sized>(&mut self, ptr: *const T) -> usize {
        let raw = ptr as *const () as usize;
        if self.visited.insert(raw) {
            unsafe { (&*ptr).inline_size() + (&*ptr).extra_heap_size(self) }
        } else {
            0
        }
    }
}

pub trait MemorySized {
    fn inline_size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn extra_heap_size(&self, _meter: &mut MemoryMeter) -> usize {
        0
    }

    fn deep_size(&self) -> usize
    where
        Self: Sized,
    {
        MemoryMeter::measure_root(self)
    }

    fn measure_with(&self, meter: &mut MemoryMeter) -> usize {
        meter.measure(self)
    }
}

macro_rules! impl_memory_for_primitives {
    ($($t:ty),* $(,)?) => {
        $(impl MemorySized for $t {})*
    };
}

impl_memory_for_primitives!(u8, u16, u32, u64, usize, i32, i64, f32, f64, bool, char);

impl MemorySized for String {
    fn inline_size(&self) -> usize {
        std::mem::size_of::<String>()
    }

    fn extra_heap_size(&self, _meter: &mut MemoryMeter) -> usize {
        self.capacity()
    }
}

impl<T: MemorySized> MemorySized for Vec<T> {
    fn inline_size(&self) -> usize {
        std::mem::size_of::<Vec<T>>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        let mut total = self.capacity() * std::mem::size_of::<T>();
        for item in self.iter() {
            total += meter.measure_field(item);
        }
        total
    }
}

impl<T: MemorySized> MemorySized for Option<T> {
    fn inline_size(&self) -> usize {
        std::mem::size_of::<Option<T>>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        match self {
            Some(value) => meter.measure_field(value),
            None => 0,
        }
    }
}

impl<T: MemorySized + ?Sized> MemorySized for Box<T> {
    fn inline_size(&self) -> usize {
        std::mem::size_of::<Box<T>>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        self.as_ref().measure_with(meter)
    }
}

impl<T: MemorySized + ?Sized> MemorySized for Rc<T> {
    fn inline_size(&self) -> usize {
        std::mem::size_of::<Rc<T>>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        unsafe { meter.measure_shared(Rc::as_ptr(self)) }
    }
}

impl<T: MemorySized + ?Sized> MemorySized for Arc<T> {
    fn inline_size(&self) -> usize {
        std::mem::size_of::<Arc<T>>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        unsafe { meter.measure_shared(Arc::as_ptr(self)) }
    }
}

impl<T: MemorySized + ?Sized> MemorySized for RefCell<T> {
    fn inline_size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        let borrowed = self.borrow();
        let extra = meter.measure_field(&*borrowed);
        drop(borrowed);
        extra
    }
}

impl<K, V, S> MemorySized for HashMap<K, V, S>
where
    K: MemorySized,
    V: MemorySized,
    S: std::hash::BuildHasher,
{
    fn inline_size(&self) -> usize {
        std::mem::size_of::<HashMap<K, V, S>>()
    }

    fn extra_heap_size(&self, meter: &mut MemoryMeter) -> usize {
        let mut total = self.capacity() * std::mem::size_of::<(K, V)>();
        for (k, v) in self.iter() {
            total += meter.measure_field(k);
            total += meter.measure_field(v);
        }
        total
    }
}
