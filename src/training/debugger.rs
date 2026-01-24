use std::io::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::palette::tailwind;
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Cell, Chart, Dataset, Gauge, GraphType, Row, Table};

use crate::backend::{backend::Backend, mode::Mode};
use crate::data::tensor_data::TensorData;
use crate::shaping::shape::Shape;
use crate::training::dataset::Dataset as ClassificationDataset;
use crate::{math::element::Element, tensor::Tensor};

pub trait Debugger<'a, B: Backend> {
    fn debug(
        &mut self,
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
        &mut self,
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
        &mut self,
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

#[derive(Default)]
pub struct VizState {
    tps: Vec<(f64, f64)>,
    tns: Vec<(f64, f64)>,
    fps: Vec<(f64, f64)>,
    fns: Vec<(f64, f64)>,
    loss: Vec<(f64, f64)>,
}

pub struct VizDebugger {
    state: Arc<Mutex<VizState>>,
    ps: usize,
    ns: usize,
    n: usize,
    x_bounds: [f64; 2],
    x_labels: [String; 3],
    y_bounds: [f64; 2],
    y_labels: [String; 3],
    loss_bounds: [f64; 2],
    loss_labels: [String; 3],
    iteration_bounds: [f64; 2],
    iteration_labels: [String; 3],
    font_color: Color,
    green: Color,
    red: Color,
}

impl VizDebugger {
    pub fn new<E: Element>(d: &ClassificationDataset<E>, iterations: usize) -> VizDebugger {
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

        let (ps, ns) =
            d.labels.iter().fold(
                (0, 0),
                |(ps, ns), y| if *y == 1 { (ps + 1, ns) } else { (ps, ns + 1) },
            );

        Self {
            state: Arc::new(Mutex::new(VizState::default())),
            ps,
            ns,
            n: d.n,
            x_bounds,
            x_labels,
            y_bounds,
            y_labels,
            loss_bounds,
            loss_labels,
            iteration_bounds,
            iteration_labels,
            font_color: tailwind::SLATE.c200,
            green: tailwind::EMERALD.c600,
            red: tailwind::ROSE.c700,
        }
    }

    pub fn run(&mut self) -> Result<(), Error> {
        let mut terminal = ratatui::init();
        let tick_rate = Duration::from_millis(250);
        let mut last_tick = Instant::now();
        loop {
            terminal.draw(|frame| self.draw(frame))?;
            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)?
                && let Event::Key(key) = event::read()?
                && key.code == KeyCode::Char('q')
            {
                break;
            }

            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }
        }

        ratatui::restore();

        Ok(())
    }

    pub fn draw(&self, frame: &mut Frame) {
        let horizontal =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]);
        let vertical = Layout::vertical([
            Constraint::Percentage(50),
            Constraint::Percentage(35),
            Constraint::Percentage(15),
        ]);
        let [scatter, right] = horizontal.areas(frame.area());
        let [loss, matrix, progress] = vertical.areas(right);
        self.render_scatter(frame, scatter);
        self.render_loss(frame, loss);
        self.render_matrix(frame, matrix);
        self.render_progress(frame, progress);
    }

    fn render_progress(&self, frame: &mut Frame, area: Rect) {
        let state = self.state.lock().unwrap();
        let iteration = state.loss.len();
        let max_iteration = self.iteration_bounds[1];

        let g = Gauge::default()
            .block(
                Block::bordered()
                    .title(Line::from("Progress").fg(self.font_color).bold().centered()),
            )
            .gauge_style(self.green)
            .ratio(iteration as f64 / max_iteration)
            .label(Span::styled(
                format!("{}/{}", iteration, max_iteration),
                Style::new().bold().fg(self.font_color),
            ));
        frame.render_widget(g, area);
    }

    fn render_matrix(&self, frame: &mut Frame, area: Rect) {
        let state = self.state.lock().unwrap();

        let neutral_style = Style::default().fg(self.font_color);
        let green_style = Style::default().fg(self.font_color).bg(self.green);
        let red_style = Style::default().fg(self.font_color).bg(self.red);

        let header_row = [
            Cell::from(Self::lines(format!(
                "Total population = P + N = {}",
                self.n
            )))
            .style(neutral_style),
            Cell::from(Self::lines(format!(
                "Predicted positive PP = {}",
                state.tps.len() + state.fps.len()
            )))
            .style(neutral_style),
            Cell::from(Self::lines(format!(
                "Predicted negative PN = {}",
                state.fns.len() + state.tns.len()
            )))
            .style(neutral_style),
        ]
        .into_iter()
        .collect::<Row>()
        .height(5);
        let second_row = [
            Cell::from(Self::lines(format!("Actual positive P = {}", self.ps)))
                .style(neutral_style),
            Cell::from(Self::lines(format!(
                "True positive TP = {}",
                state.tps.len()
            )))
            .style(green_style),
            Cell::from(Self::lines(format!(
                "False negative FN = {}",
                state.fns.len()
            )))
            .style(red_style),
        ]
        .into_iter()
        .collect::<Row>()
        .height(5);
        let third_row = [
            Cell::from(Self::lines(format!("Actual negative N = {}", self.ns)))
                .style(neutral_style),
            Cell::from(Self::lines(format!(
                "False positive FP = {}",
                state.fps.len()
            )))
            .style(red_style),
            Cell::from(Self::lines(format!(
                "True negative TN = {}",
                state.tns.len()
            )))
            .style(green_style),
        ]
        .into_iter()
        .collect::<Row>()
        .height(5);

        let t = Table::new(
            [header_row, second_row, third_row],
            [
                Constraint::Length(32),
                Constraint::Min(28),
                Constraint::Min(28),
            ],
        )
        .block(
            Block::bordered().title(
                Line::from("Confusion matrix")
                    .fg(self.font_color)
                    .bold()
                    .centered(),
            ),
        )
        .column_spacing(0);
        frame.render_widget(t, area);
    }

    fn lines<'a>(s: String) -> Vec<Line<'a>> {
        vec![Line::default(), Line::default(), Line::from(s).centered()]
    }

    fn render_scatter(&self, frame: &mut Frame, area: Rect) {
        let state = self.state.lock().unwrap();

        let datasets = vec![
            Dataset::default()
                .name("× True Positives")
                .marker(Marker::Custom('×'))
                .graph_type(GraphType::Scatter)
                .style(Style::new().green())
                .data(&state.tps),
            Dataset::default()
                .name("× False Negatives")
                .marker(Marker::Custom('×'))
                .graph_type(GraphType::Scatter)
                .style(Style::new().light_red())
                .data(&state.fns),
            Dataset::default()
                .name("• True Negatives")
                .marker(Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::new().light_green())
                .data(&state.tns),
            Dataset::default()
                .name("• False Positives")
                .marker(Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::new().red())
                .data(&state.fps),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::bordered().title(
                    Line::from("Classification")
                        .fg(self.font_color)
                        .bold()
                        .centered(),
                ),
            )
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

    fn render_loss(&self, frame: &mut Frame, area: Rect) {
        let state = self.state.lock().unwrap();

        let datasets = vec![
            Dataset::default()
                .name("Loss")
                .marker(Marker::Braille)
                .style(Style::default().fg(Color::Yellow))
                .graph_type(GraphType::Line)
                .data(&state.loss),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::bordered().title(Line::from("Loss").fg(self.font_color).bold().centered()),
            )
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

    // ratatui only supports 2 or 3 axis labels
    fn axis_labels(min: f64, max: f64) -> [String; 3] {
        let range = max - min;
        let interval = range / 2.;
        [
            format!("{min:.2}")
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
            format!("{:.2}", min + interval)
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
            format!("{max:.2}")
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
        ]
    }
}

impl Clone for VizDebugger {
    fn clone(&self) -> Self {
        VizDebugger {
            state: Arc::clone(&self.state),
            ps: self.ps,
            ns: self.ns,
            n: self.n,
            x_bounds: self.x_bounds,
            x_labels: self.x_labels.clone(),
            y_bounds: self.y_bounds,
            y_labels: self.y_labels.clone(),
            loss_bounds: self.loss_bounds,
            loss_labels: self.loss_labels.clone(),
            iteration_bounds: self.iteration_bounds,
            iteration_labels: self.iteration_labels.clone(),
            font_color: self.font_color,
            green: self.green,
            red: self.red,
        }
    }
}

impl<'a, B: Backend> Debugger<'a, B> for VizDebugger {
    fn debug(
        &mut self,
        loss: &Tensor<'a, B>,
        labels: &Tensor<'a, B>,
        output: &Tensor<'a, B>,
        iterations: (usize, usize),
        start_time: Instant,
    ) {
        let mut state = self.state.lock().unwrap();

        let total_loss = total_loss(loss);
        state.loss.push((iterations.0 as f64, total_loss.tof()));

        let output = output.data.collect();
        let labels = labels.data.collect();

        let (tps, fps, tns, fns) = output.iter().zip(labels).fold(
            (vec![], vec![], vec![], vec![]),
            |(mut tps, mut fps, mut tns, mut fns), (p, l)| {
                let pf = p.tof();
                let lf = l.tof();
                if pf > 0.5 && lf == 1. {
                    tps.push((0.25, 0.75));
                } else if pf < 0.5 && lf == 1. {
                    fps.push((0.25, 0.25));
                } else if pf < 0.5 && lf == 0. {
                    tns.push((0.75, 0.25));
                } else if pf > 0.5 && lf == 0. {
                    fns.push((0.75, 0.75));
                }
                (tps, fps, tns, fns)
            },
        );

        state.tps = tps;
        state.fps = fps;
        state.tns = tns;
        state.fns = fns;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axis_labels() {
        let labels = VizDebugger::axis_labels(5., 10.);
        assert_eq!(vec!["5", "7.5", "10"], labels);
    }
}
