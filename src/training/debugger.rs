use std::time::Instant;

use crate::backend::{backend::Backend, mode::Mode};
use crate::shaping::shape::Shape;
use crate::{math::element::Element, tensor::Tensor};

pub trait Debugger<'a, B: Backend> {
    fn debug(
        loss: &Tensor<'a, B>,
        labels: &Tensor<'a, B>,
        output: &Tensor<'a, B>,
        iterations: (usize, usize),
        start_time: Instant,
    );
}

pub struct ChattyDebugger;
impl<'a, B: Backend> Debugger<'a, B> for ChattyDebugger {
    fn debug(
        loss: &Tensor<'a, B>,
        labels: &Tensor<'a, B>,
        output: &Tensor<'a, B>,
        (current, max): (usize, usize),
        start_time: Instant,
    ) {
        if current.is_multiple_of(10) || current == max {
            let elapsed_time = start_time.elapsed();
            let total_loss = total_loss(loss);
            let correct = correct(labels, output);
            println!(
                "iteration: {current}/{max}, elapsed time: {elapsed_time:?}, loss: {total_loss}, correct: {correct}"
            );
        }
    }
}

pub struct TerseDebugger;
impl<'a, B: Backend> Debugger<'a, B> for TerseDebugger {
    fn debug(
        loss: &Tensor<'a, B>,
        labels: &Tensor<'a, B>,
        output: &Tensor<'a, B>,
        (current, max): (usize, usize),
        start_time: Instant,
    ) {
        if current == max {
            let elapsed_time = start_time.elapsed();
            let total_loss = total_loss(loss);
            let correct = correct(labels, output);
            println!(
                "iteration: {current}/{max}, elapsed time: {elapsed_time:?}, loss: {total_loss}, correct: {correct}"
            );
        }
    }
}

fn total_loss<'a, B: Backend>(loss: &Tensor<'a, B>) -> B::Element {
    loss.clone()
        .sum(None)
        .view(&Shape::scalar(1))
        .item()
        .unwrap_or(B::Element::zero())
}

fn correct<'a, B: Backend>(labels: &Tensor<'a, B>, output: &Tensor<'a, B>) -> B::Element {
    output
        .clone()
        .gt(Tensor::from_scalar(B::Element::fromf(0.5)))
        .eq(labels.clone())
        .sum(None)
        .item()
        .unwrap()
}
