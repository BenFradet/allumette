use std::time::Instant;

use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::Line;
use ratatui::widgets::{Axis, Block, Chart, Dataset, GraphType};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};

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

#[derive(Clone, Debug)]
pub struct Point {
    x: f64,
    y: f64,
    label: u8,
    predicted: u8,
}

pub struct VizDebugger<'a> {
    points: Vec<Point>,
    loss: Vec<(usize, f64)>,
    x_bounds: (f64, f64),
    x_labels: Vec<&'a str>,
    y_bounds: (f64, f64),
    y_labels: Vec<&'a str>,
    max_iterations: usize,
    n: usize,
}

impl<'a> VizDebugger<'a> {
    pub fn new(total_iterations: usize, n: usize) -> VizDebugger<'a> {
        Self {
            points: vec![],
            loss: vec![],
            x_bounds: (0., 0.),
            x_labels: vec!["0.0"],
            y_bounds: (0., 0.),
            y_labels: vec!["0.0"],
            max_iterations: total_iterations,
            n,
        }
    }

    pub fn draw(&self, frame: &mut Frame) {
        let [scatter, line_chart] =
            Layout::horizontal([Constraint::Fill(1); 2]).areas(frame.area());
        self.render_scatter(frame, scatter);
        self.render_line_chart(frame, line_chart);
    }

    fn render_scatter(&self, frame: &mut Frame, area: Rect) {
    }

    fn render_line_chart(&self, frame: &mut Frame, area: Rect) {

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
