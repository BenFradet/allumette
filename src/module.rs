use std::collections::HashMap;

use crate::parameter::Parameter;

#[derive(Debug)]
struct Module {
    modules: HashMap<String, Module>,
    parameters: HashMap<String, Parameter>,
    training: bool,
}

impl Module {
    fn new() -> Self {
        Self {
            modules: HashMap::new(),
            parameters: HashMap::new(),
            training: true,
        }
    }

    fn modules(self) -> impl Iterator<Item = Module> {
        self.modules.into_values()
    }

    // TODO: imperative version
    fn walk<F>(&mut self, f: &mut F) -> () where F: FnMut(&mut Module) -> () {
        f(self);
        for module in self.modules.values_mut() {
            module.walk(f);
        }
    }

    fn train(&mut self) -> () {
        self.walk(&mut |module| module.training = true);
    }

    fn eval(&mut self) -> () {
        self.walk(&mut |module| module.training = false);
    }

    fn training_rec(mut self, training: bool) -> () {
        self.training = training;
        self.modules().for_each(|m| m.training_rec(training));
    }

    fn train_rec(mut self) -> () {
        self.training_rec(true);
    }

    fn eval_rec(mut self) -> () {
        self.training_rec(false);
    }
}