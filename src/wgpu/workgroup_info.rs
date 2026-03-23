use crate::shaping::shape::Shape;

/// workgroup is a collection of `size` threads that execute together and share local memory
/// each thread has a unique id, allowing it to process a specific portion of the data
/// https://medium.com/@josh.sideris/mastering-thread-calculations-in-webgpu-workgroup-size-count-and-thread-identification-6b44a87a4764
///
/// count: total number of workgroups
/// size: number of threads per workgroup
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct WorkgroupInfo {
    pub count: (usize, usize, usize),
    pub size: (usize, usize, usize),
}

const MAX_WORKGROUP_COUNT: usize = 65535;
const MAX_WORKGROUP_SIZE: usize = 256;

impl WorkgroupInfo {
    pub fn for_reduce(reduce_dim: usize, shape: &Shape) -> Self {
        let tensor_size = shape.size;
        let wg_size = MAX_WORKGROUP_SIZE.min(reduce_dim.next_power_of_two());
        WorkgroupInfo {
            count: Self::chunk(tensor_size),
            size: (wg_size, 1, 1),
        }
    }

    pub fn workgroup_size(&self) -> String {
        format!(
            "@workgroup_size({}, {}, {})",
            self.size.0, self.size.1, self.size.2
        )
    }

    pub fn workgroup_size_const(&self) -> String {
        format!("const WG_SIZE: u32 = {}u;", self.size.0)
    }

    fn chunk(count: usize) -> (usize, usize, usize) {
        if count <= MAX_WORKGROUP_COUNT {
            (count, 1, 1)
        } else if count <= MAX_WORKGROUP_COUNT * MAX_WORKGROUP_COUNT {
            let y = count.div_ceil(MAX_WORKGROUP_COUNT);
            (MAX_WORKGROUP_COUNT, y, 1)
        } else {
            let z = count.div_ceil(MAX_WORKGROUP_COUNT * MAX_WORKGROUP_COUNT);
            (MAX_WORKGROUP_COUNT, MAX_WORKGROUP_COUNT, z)
        }
    }
}

impl From<&Shape> for WorkgroupInfo {
    fn from(shape: &Shape) -> Self {
        let tensor_size = shape.size;

        if tensor_size <= MAX_WORKGROUP_SIZE {
            WorkgroupInfo {
                count: (1, 1, 1),
                size: (tensor_size.next_power_of_two(), 1, 1),
            }
        } else {
            let count = tensor_size.div_ceil(MAX_WORKGROUP_SIZE);
            WorkgroupInfo {
                count: WorkgroupInfo::chunk(count),
                size: (MAX_WORKGROUP_SIZE, 1, 1),
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
                count: (2, 1, 1),
                size: (256, 1, 1)
            },
            (&Shape::new(vec![257])).into()
        );
        assert_eq!(
            WorkgroupInfo {
                count: (1, 1, 1),
                size: (32, 1, 1)
            },
            (&Shape::new(vec![14, 2])).into()
        );
        assert_eq!(
            WorkgroupInfo {
                count: (65535, 3, 1),
                size: (256, 1, 1)
            },
            (&Shape::new(vec![195313, 256])).into()
        );
    }

    #[test]
    fn for_reduce_test() {
        // reduce [3, 2] along 0th
        assert_eq!(
            WorkgroupInfo::for_reduce(3, &Shape::new(vec![1, 2])),
            WorkgroupInfo {
                count: (2, 1, 1),
                size: (4, 1, 1)
            }
        );
    }

    #[test]
    fn chunk_z_test() {
        let count = MAX_WORKGROUP_COUNT * MAX_WORKGROUP_COUNT + 1;
        assert_eq!(
            WorkgroupInfo::chunk(count),
            (MAX_WORKGROUP_COUNT, MAX_WORKGROUP_COUNT, 2),
        );
    }
}
