use std::collections::HashMap;

use proptest::prelude::*;

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

    fn arb() -> impl Strategy<Value = Module> {
        let leaf = any::<bool>().prop_map(|training| Module {
            modules: HashMap::new(),
            parameters: HashMap::new(),
            training,
        });

        leaf.prop_recursive(4, 64, 8, |inner| {
            (
                prop::collection::hash_map(".*", inner.clone(), 0..4),
                prop::collection::hash_map(".*", Parameter::arb(), 0..4),
                any::<bool>(),
            ).prop_map(|(modules, parameters, training)| Module {
                modules,
                parameters,
                training
            })
        })
    }

    fn assert_rec<F>(self, assertion: &mut F) -> () where F: FnMut(&Module) -> () {
        assertion(&self);
        self.modules().for_each(|m| m.assert_rec(assertion));
    }
}


proptest! {

    #[test]
    fn test_train(mut module in Module::arb()) {
        module.train();
        assert!(module.training);
        module.assert_rec(&mut |m| assert!(m.training));
    }

    #[test]
    fn test_eval(mut module in Module::arb()) {
        module.eval();
        assert!(!module.training);
        module.assert_rec(&mut |m| assert!(!m.training));
    }
}