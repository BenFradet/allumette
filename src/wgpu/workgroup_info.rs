use crate::shaping::shape::Shape;

/// workgroup is a collection of `size` threads that execute together and share local memory
/// each thread has a unique id, allowing it to process a specific portion of the data
/// https://medium.com/@josh.sideris/mastering-thread-calculations-in-webgpu-workgroup-size-count-and-thread-identification-6b44a87a4764
///
/// count: total number of workgroups
/// size: number of threads per workgroup
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct WorkgroupInfo {
    pub count: usize,
    pub size: usize,
}

impl WorkgroupInfo {
    pub fn workgroup_size(&self) -> String {
        format!("@workgroup_size({})", self.size)
    }
}

const MAX_WORKGROUP_COUNT: usize = 65535;
const MAX_WORKGROUP_SIZE: usize = 256;

impl From<&Shape> for WorkgroupInfo {
    fn from(shape: &Shape) -> Self {
        let tensor_size = shape.size;

        // TODO: None if tensor size > SIZE * COUNT
        if tensor_size <= MAX_WORKGROUP_SIZE {
            WorkgroupInfo {
                count: 1,
                size: tensor_size.next_power_of_two(),
            }
        } else {
            let count = (tensor_size + MAX_WORKGROUP_SIZE - 1) / MAX_WORKGROUP_SIZE;
            WorkgroupInfo {
                count,
                size: MAX_WORKGROUP_SIZE,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_shape_test() {
        assert_eq!(
            WorkgroupInfo {
                count: 2,
                size: 256
            },
            (&Shape::new(vec![257])).into()
        );
        assert_eq!(
            WorkgroupInfo { count: 1, size: 32 },
            (&Shape::new(vec![14, 2])).into()
        );
    }
}
