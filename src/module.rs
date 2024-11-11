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

    fn module_values(self) -> impl Iterator<Item = Module> {
        self.modules.into_values()
    }

    fn fold_rec<A, F>(&self, z: A, f: F) -> A where F: Fn(A, &Self) -> A + Copy {
        let mut acc = f(z, self);
        for module in self.modules.values() {
            acc = module.fold_rec(acc, f);
        };
        acc
    }

    fn fold<A, F>(&self, z: A, f: F) -> A where F: Fn(A, (&Self, String)) -> A {
        let mut stack: Vec<(&Module, String)> = vec![(self, "".to_string())];
        let mut res = z;
        while let Some((module, name)) = stack.pop() {
            res = f(res, (module, name.to_string()));
            for (name, module) in self.modules.iter() {
                stack.push((&module, name.to_string()));
            }
        }
        res
    }

    fn named_parameters(&self) -> impl Iterator<Item = (String, Parameter)> {
        self.fold((vec![], "".to_string()), |(acc, prefix), (module, module_name)| {
            let new_prefix = prefix + &module_name;
            let params = module
                .parameters
                .iter()
                .map(|(k, v)| (new_prefix.clone() + &k, v.clone()))
                .collect();
            ([acc, params].concat(), new_prefix)
        }).0.into_iter()
    }

    fn walk_rec<F>(&mut self, f: &mut F) -> () where F: FnMut(&mut Self) -> () {
        f(self);
        for module in self.modules.values_mut() {
            module.walk_rec(f);
        }
    }

    fn walk<F>(&mut self, mut f: F) -> () where F: FnMut(&mut Self) -> () {
        let mut stack = vec![self];
        while let Some(m) = stack.pop() {
            f(m);
            for module in m.modules.values_mut() {
                stack.push(module);
            }
        }
    }

    fn train(&mut self) -> () {
        self.walk(|module| module.training = true);
    }

    fn eval(&mut self) -> () {
        self.walk(|module| module.training = false);
    }

    fn arb() -> impl Strategy<Value = Module> {
        let leaf = any::<bool>().prop_map(|training| Module {
            modules: HashMap::new(),
            parameters: HashMap::new(),
            training,
        });

        leaf.prop_recursive(4, 64, 8, |inner| {
            (
                prop::collection::hash_map(".*", inner.clone(), 2),
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
        self.module_values().for_each(|m| m.assert_rec(assertion));
    }
}


proptest! {

    #[test]
    fn test_train(mut module in Module::arb()) {
        module.train();
        module.assert_rec(&mut |m| assert!(m.training));
    }

    #[test]
    fn test_eval(mut module in Module::arb()) {
        module.eval();
        module.assert_rec(&mut |m| assert!(!m.training));
    }
}