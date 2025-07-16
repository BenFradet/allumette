use std::collections::{HashMap, VecDeque};

use super::parameter::Parameter;

#[derive(Clone, Debug)]
pub struct Module<'a, A> {
    children: HashMap<&'a str, Module<'a, A>>,
    pub parameters: HashMap<&'a str, Parameter<'a, A>>,
    training: bool,
}

impl<'a, A: Clone> Module<'a, A> {
    pub fn add_parameter(mut self, param: Parameter<'a, A>) -> Self {
        self.parameters.insert(param.name, param);
        self
    }

    pub fn add_param(&mut self, param: Parameter<'a, A>) {
        self.parameters.insert(param.name, param);
    }

    pub fn add_child(&mut self, name: &'a str, module: Module<'a, A>) {
        self.children.insert(name, module);
    }

    fn children_values(self) -> impl Iterator<Item = Module<'a, A>> {
        self.children.into_values()
    }

    pub fn parameters(&self) -> impl Iterator<Item = Parameter<'a, A>> {
        self.fold_rec(vec![], |acc, module| {
            let params = module.parameters.values().cloned().collect();
            [acc, params].concat()
        })
        .into_iter()
    }

    fn named_parameters(&self) -> impl Iterator<Item = (String, Parameter<'a, A>)> {
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

    fn fold_rec<B, F>(&self, z: B, f: F) -> B
    where
        F: Fn(B, &Self) -> B + Copy,
    {
        let mut acc = f(z, self);
        for module in self.children.values() {
            acc = module.fold_rec(acc, f);
        }
        acc
    }

    fn fold<B, F>(&self, z: B, f: F) -> B
    where
        F: Fn(B, (&Self, String, u32)) -> B,
    {
        let mut stack: Vec<(&Module<'a, A>, String, u32)> = vec![(self, "".to_string(), 0)];
        let mut res = z;
        while let Some((module, name, depth)) = stack.pop() {
            res = f(res, (module, name.to_string(), depth));
            for (name, module) in module.children.iter() {
                stack.push((module, name.to_string(), depth + 1));
            }
        }
        res
    }

    fn fold_bf<B, F>(&self, z: B, f: F) -> B
    where
        F: Fn(B, (&Self, String)) -> B,
    {
        let mut queue: VecDeque<(&Module<'a, A>, String)> =
            VecDeque::from([(self, "".to_string())]);
        let mut res = z;
        while let Some((module, name)) = queue.pop_front() {
            res = f(res, (module, name.to_string()));
            for (name, module) in module.children.iter() {
                queue.push_back((module, name.to_string()));
            }
        }
        res
    }

    fn walk_rec<F>(&mut self, f: &mut F)
    where
        F: FnMut(&mut Self),
    {
        f(self);
        for module in self.children.values_mut() {
            module.walk_rec(f);
        }
    }

    fn walk<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self),
    {
        let mut stack = vec![self];
        while let Some(m) = stack.pop() {
            f(m);
            for module in m.children.values_mut() {
                stack.push(module);
            }
        }
    }

    fn train(&mut self) {
        self.walk(|module| module.training = true);
    }

    fn eval(&mut self) {
        self.walk(|module| module.training = false);
    }

    fn assert_rec<F>(self, assertion: &mut F)
    where
        F: FnMut(&Module<'a, A>),
    {
        assertion(&self);
        self.children_values().for_each(|m| m.assert_rec(assertion));
    }
}
