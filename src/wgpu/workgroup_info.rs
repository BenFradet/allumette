/// workgroup is a collection of `size` threads that execute together and share local memory
/// each thread has a unique id, allowing it to process a specific portion of the data
/// https://medium.com/@josh.sideris/mastering-thread-calculations-in-webgpu-workgroup-size-count-and-thread-identification-6b44a87a4764
///
/// count: total number of workgroups
/// size: number of threads per workgroup
struct WorkgroupInfo {
    count: usize,
    size: usize,
}
