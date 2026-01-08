use std::time::Instant;

use crate::backend::{backend::Backend, backend_type::BackendType};
use crate::shaping::shape::Shape;
use crate::{
    math::element::Element, tensor::Tensor, util::unsafe_usize_convert::UnsafeUsizeConvert,
};

pub trait Debugger<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> {
    fn debug(
        loss: &Tensor<E, BT, T>,
        labels: &Tensor<E, BT, T>,
        output: &Tensor<E, BT, T>,
        iterations: (usize, usize),
        start_time: Instant,
    );
}

pub struct ChattyDebugger;
impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Debugger<E, BT, T>
    for ChattyDebugger
{
    fn debug(
        loss: &Tensor<E, BT, T>,
        labels: &Tensor<E, BT, T>,
        output: &Tensor<E, BT, T>,
        (current, max): (usize, usize),
        start_time: Instant,
    ) {
        if current.is_multiple_of(10) || current == max {
            let elapsed_time = start_time.elapsed();
            let total_loss = total_loss(loss);
            let correct = correct(labels, output);
            println!("iteration: {current}/{max}, elapsed time: {elapsed_time:?}, loss: {total_loss}, correct: {correct}");
        }
    }
}

pub struct TerseDebugger;
impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Debugger<E, BT, T>
    for TerseDebugger
{
    fn debug(
        loss: &Tensor<E, BT, T>,
        labels: &Tensor<E, BT, T>,
        output: &Tensor<E, BT, T>,
        (current, max): (usize, usize),
        start_time: Instant,
    ) {
        if current == max {
            let elapsed_time = start_time.elapsed();
            let total_loss = total_loss(loss);
            let correct = correct(labels, output);
            println!("iteration: {current}/{max}, elapsed time: {elapsed_time:?}, loss: {total_loss}, correct: {correct}");
        }
    }
}

fn total_loss<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>>(
    loss: &Tensor<E, BT, T>,
) -> E {
    loss.clone()
        .sum(None)
        .view(&Shape::scalar(1))
        .item()
        .unwrap_or(E::zero())
}

fn correct<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>>(
    labels: &Tensor<E, BT, T>,
    output: &Tensor<E, BT, T>,
) -> E {
    output
        .clone()
        .gt(Tensor::from_scalar(E::fromf(0.5)))
        .eq(labels.clone())
        .sum(None)
        .item()
        .unwrap()
}
