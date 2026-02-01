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

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->
```rust
struct Tensor {
    data: Vec<f64>,
    shape: Shape,
    strides: Strides,
}
```
<!-- pause -->

<!-- column: 1 -->
```typst +render +width:100%
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
<!-- pause -->

<!-- column: 0 -->
```rust
struct Shape {
    data: Vec<usize>,
}
```

<!-- pause -->

```rust
struct Strides {
    data: Vec<usize>,
}
```

---

What's a tensor?
===

<!-- column_layout: [1, 2] -->

<!-- column: 0 -->
<!-- newlines: 8 -->
... you will remember there are rules
