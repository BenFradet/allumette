---
theme:
  name: catppuccin-macchiato
  override:
    default:
      margin:
        percent: 1
    typst:
      colors:
        background: cad3f500
        foreground: cad3f5
    footer:
      style: template
      left:
        image: img/logo.png
      #center: '**allumette**'
      right: "{current_slide} / {total_slides}"
      height: 3
    code:
      padding:
        vertical: 0
        horizontal: 0
      minimum_margin:
        percent: 0
options:
  end_slide_shorthand: true
---

<!-- newlines: 12 -->

<!-- column_layout: [1, 3] -->

<!-- column: 0 -->

![](img/logo.png)

<!-- column: 1 -->
<!-- newlines: 1 -->

<span style="color: #ed8796">**allumette**</span>

<span style="color: #f5a97f">a toy tensor library written in Rust</span>

<span style="color: #eed49f">Ben Fradet</span>

<!-- no_footer -->

---

Summary
===

<!-- newlines: 5 -->
# What's a tensor?
# What can we do with a tensor?
# etc

---

What's a tensor?
===

if you recall your algebra classes...
<!-- newlines: 1 -->
![image:width:70%](img/tensor.png)
<!-- alignment: center -->
credit: Cmglee, GNU FDL

<!-- newlines: 1 -->
<!-- alignment: center -->
multi-dimensional array of arbitrary dimensions

---

What does a tensor look like in rust?
===

```typst +render +width:40%
#set table(
  stroke: none,
  inset: -1pt,
)

#let cell(angle: 170deg, dx: 0em, dy: 0em) = {
  stack(
    box(width: 100%, height: 100%),

    place(
      dx: dx,
      dy: dy,
      align(
        center,
        rotate(angle, line(length: 90%, stroke: (dash: "dashed", paint: rgb("#5b6078")))),
      ),
    )
  )
}

#table(
  columns: 4,
  [], [], [#cell()], [$mat(delim: "[", 41, 42; 43, 44)$],
  [], [#cell()], [$mat(delim: "[", 31, 32; 33, 34)$], [#cell(dy: 1em, angle: 170deg)],
  [#cell()], [$mat(delim: "[", 21, 22; 23, 24)$], [#cell(angle: 170deg, dy: 1em)], [],
  [$mat(delim: "[", 11, 12; 13, 14)$], [#cell(dy: 1em)], [], [],
)
```
<!-- newlines: 1 -->
<!-- pause -->

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
```rust +no_background
struct Tensor {
    data: Vec<f64>,
    shape: Shape,
    strides: Strides,
}
```
<!-- pause -->

<!-- column: 1 -->
```rust +no_background
let data = vec![
    11., 12., 13., 14.,
    21., 22., 23., 24.,
    31., 32., 33., 34.,
    41., 42., 43., 44.
];
```
<!-- pause -->

<!-- column: 0 -->
<!-- newlines: 1 -->
```rust +no_background
struct Shape {
    data: Vec<usize>,
}
```
<!-- pause -->

<!-- column: 1 -->
```rust +no_background
let shape = Shape::new(vec![4, 2, 2]);
```
<!-- pause -->

<!-- column: 0 -->
```rust +no_background
struct Strides {
    data: Vec<usize>,
}
```
<!-- pause -->

<!-- column: 1 -->
<!-- newlines: 2 -->
```rust +no_background
let strides = Strides::new(vec![4, 2, 1]);
```
<!-- pause -->

<!-- reset_layout -->
<!-- alignment: center -->
- transposition (permutation)
- adding / removing dimensions (viewing)

=> don't require touching data

---

Summary
===

<!-- newlines: 5 -->
# What's a tensor?
# What can we do with a tensor?

---

What can be done with a tensor?
===

<!-- newlines: 5 -->
<!-- column_layout: [3, 2] -->

<!-- column: 0 -->
```rust +no_background
pub trait Ops<E: Element> {

    fn map<F: Fn(E) -> E>(&self, f: F) -> Self;
```
<!-- pause -->

<!-- column: 1 -->
<!-- newlines: 1 -->
```typst +render +width:80%
$\{\ln(x), e^x, -x, frac(1, x), ...\}$
```
<!-- pause -->

<!-- column: 0 -->
```rust +no_background

    fn zip<F: Fn(E, E) -> E>(
        &self, other: &Self, f: F
    ) -> Option<Self>;
```
<!-- pause -->

<!-- column: 1 -->
```typst +render +width:80%
$\{x + y, x dot y, x = y, ...\}$
```
<!-- pause -->

<!-- column: 0 -->
```rust +no_background

    fn reduce<F: Fn(E, E) -> E>(
        &self,
        f: F,
        dim: usize,
        zero: E,
    ) -> Option<Self>;
```
<!-- pause -->

<!-- column: 1 -->
<!-- newlines: 2 -->
```typst +render +width:80%
$\{sum(x), product(x)\}$
```
<!-- pause -->

<!-- column: 0 -->
```rust +no_background

    fn matmul(
        &self, other: &Self
    ) -> Option<Self>;

}
```
---

Map
===

<!-- newlines: 6 -->
```rust +no_background {all|4-5|6-8|9-13|all}
fn map<F: Fn(f64) -> f64>(
    &self, f: F
) -> Self {
    let len = self.size();
    let mut out = vec![0.; len];
    for (i, d) in self.data.iter().enumerate() {
        out[i] = f(*d);
    }
    Self {
        data: out,
        shape: self.shape.clone(),
        strides: self.strides.clone(),
    }
}
```
---

Map - parallel using rayon
===

<!-- newlines: 4 -->
<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
```rust +no_background {all|1|5|7|4-7|9|all}
fn map<F: Fn(f64) -> f64 + Sync>(
    &self, f: F
) -> Self {
    let out: Vec<_> = self.data
        .par_iter()
        .map(|d| f(*d))
        .collect();
    Self {
        data: Arc::new(out),
        shape: self.shape.clone(),
        strides: self.strides.clone(),
    }
}
```
<!-- pause -->

<!-- column: 1 -->
`Sync` => safe to share references between threads
<!-- pause -->
<!-- newlines: 5 -->
`Arc`  => thread-safe reference-counting pointer
<!-- pause -->

<!-- reset_layout -->
<!-- newlines: 1 -->
```rust +no_background
struct Tensor {
    data: Arc<Vec<f64>>,
    shape: Shape,
    strides: Strides,
}
```
---

Map - gpu using wgpu and wgsl
===

<!-- newlines: 4 -->
```rust +no_background {all|1-4|6-8|10|11|12|13|14|all}
@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

fn neg(in: f32) -> f32 {
    return -in;
} // etc.

@compute
@workgroup_size(x, y, z)
fn call(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    output[i] = replace_me(input[i]);
}
```

---

Map - orchestrating gpu code
===

```rust +no_background {all|1|2|4-5|7-10|4,12|14|16|all}
fn map(&self, f: &'static str) -> Self {
    let output_buffer = create_output_buffer(self.shape.gpu_byte_size());
    
    let workgroups = (&self.shape).into();
    let pipeline = get_or_create_pipeline(f, workgroups.size);

    let bind_group = create_bind_group(
        &[&self.buffer, &output_buffer],
        &pipeline.get_bind_group_layout(0),
    );

    let command = encode_command(&workgroups.count, &pipeline, &bind_group);

    submit_command(command);

    self.with_buffer(output_buffer)
}
```

<!-- pause -->
<!-- newlines: 1 -->
```rust +no_background
struct Tensor<'a> {
    buffer: Arc<Buffer>,
    shape: Shape,
    strides: Strides,
    context: &'a WgpuContext,
}
```
---

Map - gpu parallelism
===

<!-- alignment: center -->
workgroups are collections of threads that execute together and share local memory
<!-- pause -->

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
![](img/wg.png)
<!-- alignment: center -->
credit: CubeCL, Apache 2.0
<!-- pause -->

<!-- column: 1 -->
identifiers
```typst +render +width:100%
$vec("global_inv_id.x", "global_inv_id.y", "global_inv_id.z") = vec("local_inv_id.x" times "workgroup_id.x", "local_inv_id.y" times "workgroup_id.y", "local_inv_id.z" times "workgroup_id.z")$
```
<!-- pause -->
total invocations
```typst +render +width:80%
$vec("workgroup_size.x", "workgroup_size.y", "workgroup_size.z") dot vec("num_workgroups.x", "num_workgroups.y", "num_workgroups.z")$
```
<!-- pause -->

<!-- column: 0 -->
<!-- newlines: 1 -->
```rust +no_background {all|2|6-7|8|all}
@compute
@workgroup_size(wsx, wsy, wsz)
fn call(@builtin(global_invocation_id)
    id: vec3<u32>
) {
    // (id.x, id.y, id.z) unique thread id
    let i = id.x; 
    output[i] = input[i];
}
```
<!-- pause -->

<!-- column: 1 -->
<!-- newlines: 2 -->
```rust +no_background {all|6|all}
// encode_command explained
let mut encoder = create_command_encoder();
let mut pass = encoder.begin_compute_pass();
pass.set_pipeline(pipeline);
pass.set_bind_group(0, Some(bind_group));
pass.dispatch_workgroups(nwx, nwy, nwz);
encoder.finish()
```
---

Summary
===

<!-- newlines: 5 -->
# What's a tensor?
# What can we do with a tensor?
## Map
## Zip

---

Zip - broadcasting
===

<!-- alignment: center -->
...you'll remember there are rules for element-wise operations...
<!-- pause -->
<!-- newlines: 1 -->

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
<!-- alignment: center -->
<span style="color:#a6da95">do's 👍</span>
<!-- pause -->

```typst +render +width:60%
$
1
+
underbrace(
  mat(
    1, ..., 2;
    dots.v, dots.down, dots.v;
    3, ..., 4;
  ),
  n,
) lr(size: #3em, brace.r) m
=
underbrace(
  mat(
    2, ..., 3;
    dots.v, dots.down, dots.v;
    4, ..., 5;
  ),
  n,
) lr(size: #3em, brace.r) m
$
```
<!-- pause -->
```typst +render +width:70%
$
m lr(size: #3em, brace.l) vec(
  1, dots.v, 2
) 
+
underbrace(
  mat(
    1, ..., 2;
    dots.v, dots.down, dots.v;
    3, ..., 4;
  ),
  n,
) lr(size: #3em, brace.r) m
=
underbrace(
  mat(
   2, ..., 3;
   dots.v, dots.down, dots.v;
   5, ..., 6;
  ),
  n,
) lr(size: #3em, brace.r) m
$
```
<!-- pause -->
```typst +render +width:70%
$
underbrace(
  (1 ... 2),
  n
) 
+
underbrace(
  mat(
    1, ..., 2;
    dots.v, dots.down, dots.v;
    3, ..., 4;
  ),
  n,
) lr(size: #3em, brace.r) m
=
underbrace(
  mat(
   2, ..., 4;
   dots.v, dots.down, dots.v;
   4, ..., 6;
  ),
  n,
) lr(size: #3em, brace.r) m
$
```
<!-- pause -->
```typst +render +width:70%
$
m lr(size: #3em, brace.l) underbrace(
  mat(
    1, ..., 2;
    dots.v, dots.down, dots.v;
    3, ..., 4;
  ),
  n,
) 
+
underbrace(
  mat(
    1, ..., 2;
    dots.v, dots.down, dots.v;
    3, ..., 4;
  ),
  n,
) lr(size: #3em, brace.r) m
=
underbrace(
  mat(
   2, ..., 4;
   dots.v, dots.down, dots.v;
   6, ..., 8;
  ),
  n,
) lr(size: #3em, brace.r) m
$
```
<!-- pause -->

<!-- column: 1 -->
<!-- alignment: center -->
<span style="color:#ed8796">dont's 👎</span>

<!-- pause -->
```typst +render +width:80%
$
p != m, #h(0.5em)
p lr(size: #3em, brace.l) vec(
  a_1, dots.v, a_p
) 
+
underbrace(
  mat(
    b_11, ..., b(1n);
    dots.v, dots.down, dots.v;
    b_(m 1), ..., b_(m n);
  ),
  n,
) lr(size: #3em, brace.r) m
$
```
<!-- pause -->
```typst +render +width:80%
$
q != n, #h(0.5em)
underbrace(
  (a_1 ... a_q),
  q
) 
+
underbrace(
  mat(
    b_11, ..., b_(1n);
    dots.v, dots.down, dots.v;
    b_(m 1), ..., b_(m n);
  ),
  n,
) lr(size: #3em, brace.r) m
$
```
<!-- pause -->
```typst +render +width:100%
$
p != m, #h(0.5em) q != n, #h(0.5em)
p lr(size: #3em, brace.l) underbrace(
  mat(
    a_11, ..., a_(1q);
    dots.v, dots.down, dots.v;
    a_(p 1), ..., a_(p q);
  ),
  q,
) 
+
underbrace(
  mat(
    b_11, ..., b_(1n);
    dots.v, dots.down, dots.v;
    b_(m 1), ..., b_(m n);
  ),
  n,
) lr(size: #3em, brace.r) m
$
```

---

Zip - broadcasting cont'd
===

<!-- column_layout: [6, 3] -->

<!-- column: 0 -->
<!-- newlines: 3 -->
```rust +no_background {all|1|2-3|5|7|8-9|11-16|19|all}
fn broadcast(&self, b: &Shape) -> Option<Shape> {
    let (n, m) = (self.len(), b.len());
    let max = m.max(n);

    let mut out = Vec::with_capacity(max);

    for i in (0..max).rev() {
        let si = if i < n { self[n - 1 - i] } else { 1 };
        let bi = if i < m { b[m - 1 - i] } else { 1 };

        match (si, bi) {
            (1, b) => out.push(b),
            (s, 1) => out.push(s),
            (s, b) if s == b => out.push(s),
            _ => return None,
        }
    }

    Some(Shape::new(out))
}
```
<!-- pause -->

<!-- column: 1 -->
```typst +render +width:70%
$vec(1) #h(0.5em) "bc" #h(0.5em) vec(2, 3) = vec(2, 3)$
```
<!-- pause -->
```typst +render +width:70%
$vec(5) #h(0.5em) "bc" #h(0.5em) vec(2, 5) = vec(2, 5)$
```
<!-- pause -->
```typst +render +width:85%
$vec(2, 3, 1) #h(0.5em) "bc" #h(0.5em) vec(7, 2, 3, 5) = vec(7, 2, 3, 5)$
```
<!-- pause -->
```typst +render +width:55%
$vec(5) #h(0.5em) #strike(stroke: 1pt + red)[bc] #h(0.5em) vec(5, 2)$
```
<!-- pause -->
```typst +render +width:65%
$vec(5, 7, 5, 1) #h(0.5em) #strike(stroke: 1pt + red)[bc] #h(0.5em) vec(1, 5, 1, 5)$
```

---

Zip - impl
===

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
```rust +no_background {all|1-5|6-10|11|13-14|16-26|17|18-19|20-21|22-23|24|25|28|all}
fn zip<F: Fn(f64, f64) -> f64>(
    &self,
    other: &Self,
    f: F
) -> Option<Self> {
    let shape = if self.shape == other.shape {
        self.shape.clone()
    } else {
        self.shape.broadcast(&other.shape)?
    };
    let strides: Strides = (&shape).into();

    let len = shape.size;
    let mut out = vec![0.; len];

    for i in 0..len {
        let idx = strides.idx(i);
        let idxa = idx.broadcast(&self.shape);
        let idxb = idx.broadcast(&other.shape);
        let posa = self.strides.position(&idxa);
        let posb = other.strides.position(&idxb);
        let va = self.data[posa];
        let vb = other.data[posb];
        let pos = strides.position(&idx);
        out[pos] = f(va, vb);
    }

    Some(Self::new(out, shape, strides))
}
```
<!-- pause -->

<!-- column: 1 -->
```typst +render +width:80%
$
    1 lr(size: #1em, brace.l) underbrace((1 #h(0.5em) 2), 2) +
    underbrace(vec(3, 4), 1) lr(size: #2em, brace.r) 2 =
    mat(4, 5; 5, 6)
$
```
<!-- pause -->
```rust +no_background
[2, 2]
```
<!-- pause -->
```rust +no_background
[2, 1]
```
<!-- pause -->
<!-- newlines: 1 -->
```rust +no_background
out = [0, 0, 0, 0]
```
<!-- pause -->
```rust +no_background
i = 3
idx = [1, 1]  // 1D => nD: 3 / 2 => (3 % 2) / 1
idxa = [0, 1] // [1, 1] bc [1, 2]
idxb = [1, 0] // [1, 1] bc [2, 1]
posa = 1      // nD => 1D: [1, 2] pos [0, 1]
posb = 1      // nD => 1D: [2, 1] pos [1, 0]
va = 2
vb = 4
pos = 3       // nD => 1D: [2, 1] pos [1, 1]
out[3] = 2 + 4 = 6
```

---

Summary
===

<!-- newlines: 5 -->
# What's a tensor?
# What can we do with a tensor?
## Map
## Zip
## Reduce

---

Reduce
===

<!-- newlines: 3 -->
<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
```typst +render +width:80%
$
    sum_(d=0) mat(1, 2, 3; 4, 5, 6) = #h(0.5em) ?
$
```
<!-- pause -->
```typst +render +width:50%
$
    (5 #h(0.5em) 7 #h(0.5em) 9)
$
```
<!-- pause -->

<!-- column: 1 -->
```typst +render +width:80%
$
    sum_(d=1) mat(1, 2, 3; 4, 5, 6) = #h(0.5em) ?
$
```
<!-- pause -->
```typst +render +width:40%
$
    vec(6, 15)
$
```

---

Reduce - impl
===

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
```rust +no_background {all|1-6|7-9|10|12-13|15-25|16|17|18-24|19|20|21-22|23|all}
fn reduce<F: Fn(f64, f64) -> f64>(
    &self,
    f: F,
    dim: usize,
    zero: f64,
) -> Self {
    let mut shape_data = self.shape.to_vec();
    shape_data[dim] = 1;
    let shape = Shape::new(shape_data);
    let strides: Strides = (&shape).into();

    let len = shape.size;
    let mut out = vec![zero; len];

    for i in 0..len {
        let mut idx = strides.idx(i);
        let out_pos = strides.position(&idx);
        for j in 0..self.shape[dim] {
            idx[dim] = j;
            let pos = self.strides.position(&idx);
            let v_self = self.data[pos];
            out[out_pos] = f(out[out_pos], v_self);
        }
    }
    Self::new(out, shape, strides)
}
```

<!-- pause -->

<!-- column: 1 -->
```typst +render +width:55%
$
    product_(d=1) mat(1, 2, 3; 4, 5, 6; 7, 8, 9) = vec(6, 120, 504)
$
```
<!-- pause -->
```rust +no_background
[3, 1]
[1, 1]
```
<!-- pause -->
<!-- newlines: 1 -->
```rust +no_background
out = [1, 1, 1]
```
<!-- pause -->
```rust +no_background
i = 2
idx = [2, 0]      // 1D => nD: 2 / 1 => (2 % 1) / 1
out_pos = 2       // [1, 1] pos [2, 0]
j = {0, 1, 2}
idx = {[2, 0], [2, 1], [2, 2]}
pos = {6, 7, 8}   // nD => 1D: [3, 1] pos idx
v_self = {7, 8, 9}
out[2] = {1 * 7, 7 * 8, 56 * 9}
```

---

Summary
===

<!-- newlines: 5 -->
# What's a tensor?
# What can we do with a tensor?
## Map
## Zip
## Reduce
## Matmul

---

Matmul
===

<!-- alignment: center -->
...you'll also remember there are rules for matrix multiplication...


```typst +render +width:55%
#let hlp(x) = text(fill: rgb("#f5a97f"))[$#x$]
#let hlm(x) = text(fill: rgb("#c6a0f6"))[$#x$]

#let hll(x) = {
  set text(fill: gradient.linear(rgb("#c6a0f6"), rgb("#f5a97f")))
  box($#x$)
}
#let hlr(x) = {
  set text(fill: gradient.linear(rgb("#f5a97f"), rgb("#c6a0f6")))
  box($#x$)
}

#grid(
  columns: 2,
  column-gutter: 0.5em,
  row-gutter: 0.3em,
  align: (right + bottom, left + bottom),
  [],
  $ n lr(size: #3em, brace.l) overbrace(mat(hlm(1), hlp(2); hlm(3), hlp(4); hlm(5), hlp(6)), p) $,
  $ m lr(size: #2em, brace.l) underbrace(mat(hlm(1), hlm(2), hlm(3); hlp(4), hlp(5), hlp(6)), n) $,
  $ m lr(size: #2em, brace.l) underbrace(mat(hlm(22), hll(28); hlr(49), hlp(64)), p) $,
)
```

---

Matmul - impl
===

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
<!-- newlines: 5 -->
```rust +no_background {all|5-7|9-10|12-13|15-16|17-18|all}
fn matmul(&self, rhs: &Self) -> Option<Self> {
    let lhs_rank = self.shape.len();
    let rhs_rank = rhs.shape.len();
    
    let n1 = self.shape[lhs_rank - 1];
    let n2 = rhs.shape[rhs_rank - 2];
    (n1 == n2).then_some(())?;

    let m = self.shape[lhs_rank - 2];
    let p = rhs.shape[rhs_rank - 1];

    let lhs_shape = self.shape.drop_right(2);
    let rhs_shape = rhs.shape.drop_right(2);

    let mut shape =
        lhs_shape.broadcast(&rhs_shape)?;
    shape.push(m);
    shape.push(p);
    let len = shape.size;
    let strides: Strides = (&shape).into();

    let mut out = vec![0.; len];

    // to be continued ...
```

<!-- pause -->

<!-- column: 1 -->
```typst +render +width:90%
$
m lr(size: #2em, brace.l) underbrace(mat(1, 2, 3; 4, 5, 6), n)  times
underbrace(mat(1, 2; 3, 4; 5, 6), p) lr(size: #3em, brace.r) n =
underbrace(mat(22, 28; 49, 64), p) lr(size: #2em, brace.r) m
$
```

<!-- pause -->

```rust +no_background
n1 = 3 // nb cols A
n2 = 3 // nb rows B
```
<!-- pause -->
<!-- newlines: 1 -->
```rust +no_background
m = 2 // nb rows A
p = 2 // nb cols B
```
<!-- pause -->
<!-- newlines: 4 -->
```rust +no_background
[]
[2]
[2, 2]

[2, 1]
```
<!-- pause -->
```rust +no_background
[0, 0, 0, 0]
```

---

Matmul - impl cont'd
===

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
<!-- newlines: 7 -->
```rust +no_background
    // ... continuing

    for (i, out_i) in out.iter_mut().enumerate() {

        let idx = strides.idx(i);
        let mut lhs_idx = idx.broadcast(&self.shape);
        let mut rhs_idx = idx.broadcast(&rhs.shape);

        let mut tmp = 0.;
        for pos in 0..n {
            lhs_idx[lhs_rank - 1] = pos;
            rhs_idx[rhs_rank - 2] = pos;
            let lhs_pos =
                self.strides.position(&lhs_idx);
            let rhs_pos =
                rhs.strides.position(&rhs_idx);
            tmp +=
                self.data[lhs_pos] * rhs.data[rhs_pos];
        }
        *out_i = tmp;
    }

    Some(Self::new(out, shape, strides))
}
```

<!-- pause -->

<!-- column: 1 -->
```typst +render +width:90%
$
m lr(size: #2em, brace.l) underbrace(mat(1, 2, 3; 4, 5, 6), n)  times
underbrace(mat(1, 2; 3, 4; 5, 6), p) lr(size: #3em, brace.r) n =
underbrace(mat(22, 28; 49, 64), p) lr(size: #2em, brace.r) m
$
```

<!-- pause -->

```rust +no_background
i = 3
```
<!-- pause -->
```rust +no_background
idx = [1, 1] // [2, 1] idx 3, 3 / 2 => (3 % 2) / 1
lhs_idx = [1, 1] // [1, 1] bc [2, 3]
rhs_idx = [1, 1] // [1, 1] bc [3, 2]
```
<!-- pause -->
```rust +no_background

pos = {0, 1, 2}
lhs_idx = {[1, 0], [1, 1], [1, 2]}
rhs_idx = {[0, 1], [1, 1], [2, 1]}

lhs_pos = {3, 4, 5} // nD => 1D: [3, 1] pos lhs_idx
rhs_pos = {1, 3, 5} // nD => 1D: [2, 1] pos rhs_idx

tmp = {4 * 2, 8 + 5 * 4, 28 + 6 * 6} => 64
```
