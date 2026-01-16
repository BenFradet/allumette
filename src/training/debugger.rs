use std::time::Instant;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::Line;
use ratatui::widgets::{Axis, Block, Chart, Dataset, GraphType};

use crate::backend::{backend::Backend, mode::Mode};
use crate::shaping::shape::Shape;
use crate::training::dataset::Dataset as ClassificationDataset;
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

pub struct VizDebugger {
    points: Vec<Point>,
    loss: Vec<(usize, f64)>,
    x_bounds: [f64; 2],
    x_labels: [String; 3],
    y_bounds: [f64; 2],
    y_labels: [String; 3],
    loss_bounds: [f64; 2],
    loss_labels: [String; 3],
    iteration_bounds: [f64; 2],
    iteration_labels: [String; 3],
}

impl VizDebugger {
    pub fn new<E: Element>(d: ClassificationDataset<E>, iterations: usize) -> VizDebugger {
        let (x_bounds, y_bounds) = d.features.iter().fold(
            (
                [f64::INFINITY, f64::NEG_INFINITY],
                [f64::INFINITY, f64::NEG_INFINITY],
            ),
            |([min_x, max_x], [min_y, max_y]), (x, y)| {
                let (xf, yf) = (x.tof(), y.tof());
                (
                    [min_x.min(xf), max_x.max(xf)],
                    [min_y.min(yf), max_y.max(yf)],
                )
            },
        );
        let x_labels = Self::axis_labels(x_bounds[0], x_bounds[1]);
        let y_labels = Self::axis_labels(y_bounds[0], y_bounds[1]);

        let loss_bounds = [0., d.n as f64];
        let loss_labels = Self::axis_labels(loss_bounds[0], loss_bounds[1]);
        let iteration_bounds = [0., iterations as f64];
        let iteration_labels = Self::axis_labels(iteration_bounds[0], iteration_bounds[1]);

        Self {
            points: vec![],
            loss: vec![],
            x_bounds,
            x_labels,
            y_bounds,
            y_labels,
            loss_bounds,
            loss_labels,
            iteration_bounds,
            iteration_labels,
        }
    }

    pub fn draw(&self, frame: &mut Frame) {
        let [scatter, line_chart] =
            Layout::horizontal([Constraint::Fill(1); 2]).areas(frame.area());
        self.render_scatter(frame, scatter);
        self.render_line_chart(frame, line_chart);
    }

    fn render_scatter(&self, frame: &mut Frame, area: Rect) {
        let datasets = vec![
            Dataset::default()
                .name("Correct")
                .marker(Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::new().green())
                .data(&[]),
            Dataset::default()
                .name("Incorrect")
                .marker(Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::new().red())
                .data(&[]),
        ];

        let chart = Chart::new(datasets)
            .block(Block::bordered().title(Line::from("Classification").cyan().bold().centered()))
            .x_axis(
                Axis::default()
                    .title("x")
                    .bounds(self.x_bounds)
                    .style(Style::default().fg(Color::Gray))
                    .labels(self.x_labels.clone()),
            )
            .y_axis(
                Axis::default()
                    .title("y")
                    .bounds(self.y_bounds)
                    .style(Style::default().fg(Color::Gray))
                    .labels(self.y_labels.clone()),
            )
            .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)));

        frame.render_widget(chart, area);
    }

    fn render_line_chart(&self, frame: &mut Frame, area: Rect) {
        let datasets = vec![
            Dataset::default()
                .name("Loss")
                .marker(Marker::Braille)
                .style(Style::default().fg(Color::Yellow))
                .graph_type(GraphType::Line)
                .data(&[]),
        ];

        let chart = Chart::new(datasets)
            .block(Block::bordered().title(Line::from("Loss").cyan().bold().centered()))
            .x_axis(
                Axis::default()
                    .title("iteration")
                    .bounds(self.iteration_bounds)
                    .style(Style::default().fg(Color::Gray))
                    .labels(self.iteration_labels.clone()),
            )
            .y_axis(
                Axis::default()
                    .title("loss")
                    .bounds(self.loss_bounds)
                    .style(Style::default().fg(Color::Gray))
                    .labels(self.loss_labels.clone()),
            )
            .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)));

        frame.render_widget(chart, area);
    }

    fn axis_labels(min: f64, max: f64) -> [String; 3] {
        let range = max - min;
        let interval = range / 3.;
        [
            format!("{min:.2}")
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
            format!("{:.2}", min + interval)
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
            format!("{:.2}", min + interval * 2.)
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
        ]
    }

    // ratatui only supports 2 or 3 axis labels
    //fn axis_labels(n: u8, min: f64, max: f64) -> Vec<String> {
    //    let range = max - min;
    //    let interval = range / n as f64;
    //    (0..n).map(|i| (min + i as f64 * interval).to_string()).collect()
    //}
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axis_labels() {
        let labels = VizDebugger::axis_labels(5., 10.);
        //let labels = VizDebugger::axis_labels(4, 5., 10.);
        assert_eq!(vec!["5", "6.67", "8.33"], labels);
    }
}
