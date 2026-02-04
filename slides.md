---
theme:
  name: catppuccin-macchiato
  override:
    typst:
      colors:
        background: cad3f500
        foreground: cad3f5
    footer:
      style: template
      left:
        image: img/logo.png
      center: '**allumette**'
      right: "{current_slide} / {total_slides}"
      height: 3
    code:
      padding:
        vertical: 0
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

What's a tensor?
===

<!-- column_layout: [1, 2] -->

<!-- column: 0 -->
<!-- newlines: 8 -->
<!-- font_size: 7 -->
if you recall your algebra classes...

<!-- column: 1 -->
<!-- newlines: 3 -->
![](img/tensor.png)
<!-- alignment: center -->
<!-- font_size: 1 -->
credit: Cmglee, GNU FDL

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
```rust +no_background
let tensor = Tensor { data, shape, strides };
```

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
```typst +render +width:80%
$\{\ln(x), e^x, -x, frac(1, x), ...\}$
```
<!-- pause -->

<!-- column: 0 -->
```rust +no_background

      fn zip<F: Fn(E, E) -> E>(
          &self, other: &Self, f: F) -> Option<Self>;
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
```typst +render +width:80%
$\{sum(x), product(x)\}$
```
<!-- pause -->

<!-- column: 0 -->
```rust +no_background

      fn matmul(&self, other: &Self) -> Option<Self>;
  }
```


---

What's a tensor?
===

<!-- column_layout: [1, 2] -->

<!-- column: 0 -->
<!-- newlines: 8 -->
... you will remember there are rules
