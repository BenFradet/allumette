use std::collections::{HashMap, VecDeque};

use super::parameter::Parameter;

#[derive(Debug)]
pub struct Module {
    children: HashMap<String, Module>,
    parameters: HashMap<String, Parameter>,
    training: bool,
}

impl Default for Module {
    fn default() -> Self {
        Self {
            children: HashMap::new(),
            parameters: HashMap::new(),
            training: true,
        }
    }
}

impl Module {
    pub fn add_parameter(mut self, param: Parameter) -> Self {
        let name = param.name.to_owned();
        self.parameters.insert(name, param);
        self
    }

    fn children_values(self) -> impl Iterator<Item = Module> {
        self.children.into_values()
    }

    fn parameters(&self) -> impl Iterator<Item = Parameter> {
        self.fold_rec(vec![], |acc, module| {
            let params = module.parameters.values().cloned().collect();
            [acc, params].concat()
        })
        .into_iter()
    }

    fn named_parameters(&self) -> impl Iterator<Item = (String, Parameter)> {
        fn build_prefix(prefix: String, current_mod_name: String, depth: u32) -> String {
            if depth == 1 {
                current_mod_name
            } else {
                prefix + "." + &current_mod_name
            }
        }

        fn build_p_name(prefix: &str, parameter_name: &str, depth: u32) -> String {
            if depth == 0 {
                parameter_name.to_string()
            } else {
                prefix.to_string() + "." + parameter_name
            }
        }

        self.fold(
            (vec![], "".to_string()),
            |(acc, prefix), (module, mod_name, depth)| {
                let new_prefix = build_prefix(prefix, mod_name, depth);
                let params = module
                    .parameters
                    .iter()
                    .map(|(k, v)| (build_p_name(&new_prefix, k, depth), v.clone()))
                    .collect();
                ([acc, params].concat(), new_prefix)
            },
        )
        .0
        .into_iter()
    }

    fn fold_rec<A, F>(&self, z: A, f: F) -> A
    where
        F: Fn(A, &Self) -> A + Copy,
    {
        let mut acc = f(z, self);
        for module in self.children.values() {
            acc = module.fold_rec(acc, f);
        }
        acc
    }

    fn fold<A, F>(&self, z: A, f: F) -> A
    where
        F: Fn(A, (&Self, String, u32)) -> A,
    {
        let mut stack: Vec<(&Module, String, u32)> = vec![(self, "".to_string(), 0)];
        let mut res = z;
        while let Some((module, name, depth)) = stack.pop() {
            res = f(res, (module, name.to_string(), depth));
            for (name, module) in module.children.iter() {
                stack.push((&module, name.to_string(), depth + 1));
            }
        }
        res
    }

    fn fold_bf<A, F>(&self, z: A, f: F) -> A
    where
        F: Fn(A, (&Self, String)) -> A,
    {
        let mut queue: VecDeque<(&Module, String)> = VecDeque::from([(self, "".to_string())]);
        let mut res = z;
        while let Some((module, name)) = queue.pop_front() {
            res = f(res, (module, name.to_string()));
            for (name, module) in module.children.iter() {
                queue.push_back((&module, name.to_string()));
            }
        }
        res
    }

    fn walk_rec<F>(&mut self, f: &mut F) -> ()
    where
        F: FnMut(&mut Self) -> (),
    {
        f(self);
        for module in self.children.values_mut() {
            module.walk_rec(f);
        }
    }

    fn walk<F>(&mut self, mut f: F) -> ()
    where
        F: FnMut(&mut Self) -> (),
    {
        let mut stack = vec![self];
        while let Some(m) = stack.pop() {
            f(m);
            for module in m.children.values_mut() {
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

    fn assert_rec<F>(self, assertion: &mut F) -> ()
    where
        F: FnMut(&Module) -> (),
    {
        assertion(&self);
        self.children_values().for_each(|m| m.assert_rec(assertion));
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar::scalar::Scalar;

    use super::*;

    fn test_para_a() -> Parameter {
        Parameter {
            name: "parameter_a".to_string(),
            scalar: Scalar::new(50.),
        }
    }

    fn test_para_b() -> Parameter {
        Parameter {
            name: "parameter_b".to_string(),
            scalar: Scalar::new(100.),
        }
    }

    fn test_module() -> Module {
        let para_a = test_para_a();
        let para_b = test_para_b();
        Module {
            children: HashMap::from([
                (
                    "module_a".to_string(),
                    Module {
                        children: HashMap::new(),
                        parameters: HashMap::from([
                            (para_a.name.clone(), para_a.clone()),
                            (para_b.name.clone(), para_b.clone()),
                        ]),
                        training: false,
                    },
                ),
                (
                    "module_b".to_string(),
                    Module {
                        children: HashMap::new(),
                        parameters: HashMap::from([
                            (para_a.name.clone(), para_a.clone()),
                            (para_b.name.clone(), para_b.clone()),
                        ]),
                        training: false,
                    },
                ),
            ]),
            parameters: HashMap::from([(para_a.name.clone(), para_a.clone())]),
            training: false,
        }
    }

    #[test]
    fn parameters_test() -> () {
        let module = test_module();
        let parameters: Vec<_> = module.parameters().collect();
        assert_eq!(5, parameters.len());
    }

    #[test]
    fn named_parameters_test() -> () {
        let module = test_module();
        let para_a = test_para_a();
        let para_b = test_para_b();
        let named_parameters: Vec<_> = module.named_parameters().collect();
        let expected = vec![
            (para_a.name.clone(), para_a.clone()),
            ("module_b.".to_string() + &para_b.name, para_b.clone()),
            ("module_b.".to_string() + &para_a.name, para_a.clone()),
            ("module_a.".to_string() + &para_b.name, para_b.clone()),
            ("module_a.".to_string() + &para_a.name, para_a.clone()),
        ];
        // can't have partial eq on scalar because of scalar function's rc<dyn>
        assert!(expected
            .iter()
            .all(|(en, pn)| named_parameters.iter().any(|(n, p)| {
                n == en
                    && p.name == pn.name
                    && p.scalar.derivative == pn.scalar.derivative
                    && p.scalar.v == pn.scalar.v
            })));
    }
}
