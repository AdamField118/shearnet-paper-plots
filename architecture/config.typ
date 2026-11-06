// All panels use the original drawing coordinates (x right, y up).
// A viewport maps them to paper points. Forward extends past the last axis
// tick to include the output labels; the other crops match the original axes.
#let forward-width = 811.983pt
#let views = (
  forward: (origin: (0,42), size: (forward-width,220.32pt), extent: (forward-width/4.4pt,50)),
  detail3_network: (origin: (29.8,0.8), size: (547.2pt,238.32pt), extent: (93.4,40.7)),
  detail4_reynolds: (origin: (0.5,0.5), size: (518.4pt,190.8pt), extent: (79,29)),
  details56_pooling_heads: (origin: (123.7,0.8), size: (518.4pt,379.44pt), extent: (55.6,40.7)),
)
#let colors = (
  ink: rgb("#171717"), muted: rgb("#5A6168"), line: rgb("#2D3135"),
  galaxy: rgb("#C45A15"), psf: rgb("#087C78"), orbit: rgb("#B47B13"),
  feature: rgb("#4C6FA8"), fusion: rgb("#34764B"), pool: rgb("#6652A3"), head: rgb("#A43B46"),
)
#let fills = (galaxy: rgb("#F7E3D5"), psf: rgb("#DCEFED"), orbit: rgb("#FBF1D8"),
  feature: rgb("#DFE8F5"), fusion: rgb("#E1F0E5"), pool: rgb("#EAE6F6"), head: rgb("#F5E1E4"))
#let typeface = (text: "DejaVu Serif", math: "DejaVu Math TeX Gyre")
#let fs = (group: 14pt, title: 12pt, formula: 11.2pt, body: 10.8pt, small: 9.8pt,
  tiny: 9pt, layer: 8.2pt, narrow: 7.8pt, head: 8pt, detail: 8.1pt, equation: 10.2pt)
#let pen = (frame: 1pt, box: 0.85pt, inner: 0.7pt, fine: 0.55pt,
  pad: 1.8pt, leading: 0.27em, rim: 0.8pt, detail: 0.9pt,
  fusion: 0.6pt, sheet: 0.65pt, head: 0.75pt)

// Physical dimensions keep a serial connection identical in every panel.
#let connector = (span: 16pt, stroke: 1pt, clearance: 1pt, head: "-|>", head-scale: 60%)

// Display order. Numbered steps also supply the cross-references in detail titles.
// Feature-map displays have captions but do not count as processing steps.
#let flow = (inputs: true, orbit: true, fusion: true, features: false,
  reynolds: true, maps: false, pooling: true, heads: true)
#let step(id) = {
  let steps = flow.keys().filter(k => flow.at(k))
  if id in steps {steps.position(k => k == id) + 1}
}
#let detail-title(ids, title) = {
  let nums = ids.map(step).filter(n => n != none).sorted()
  if nums.len() == 1 {[Detail #nums.first(): ]}
  if nums.len() > 1 {
    let consecutive = nums.last() - nums.first() == nums.len() - 1
    let numbers = if consecutive {[#nums.first()-#nums.last()]} else {nums.map(str).join(", ")}
    [Details #numbers: ]
  }
  title
}

// Model facts used by multiple panels. Each encoder stage carries both branch depths.
#let stages = (
  (channels: 32, galaxy: 2, psf: 1, downsample: false),
  (channels: 48, galaxy: 2, psf: 1, downsample: true),
  (channels: 64, galaxy: 1, psf: 1, downsample: true),
)
// Keep each transform with its two Reynolds signs when changing the display order.
#let orbit = (
  (op: $1$,                w1: "+", w2: "+"),
  (op: $hat(R)_90$,        w1: "-", w2: "-"),
  (op: $hat(R)_90^2$,      w1: "+", w2: "+"),
  (op: $hat(R)_90^3$,      w1: "-", w2: "-"),
  (op: $hat(P)$,           w1: "+", w2: "-"),
  (op: $hat(P)hat(R)_90$,  w1: "-", w2: "+"),
  (op: $hat(P)hat(R)_90^2$, w1: "+", w2: "-"),
  (op: $hat(P)hat(R)_90^3$, w1: "-", w2: "+"),
)
#let attention = ("attention-0.png","attention-1.png","attention-2.png","attention-3.png")
#let model = (pixels: 53, side: 13, heads: 4, ffn: 256, hidden: 128, psf-orbits: 4,
  stride: 2, kernel: 3, dilations: (1,2,4))
#let ch = stages.last().channels
#let pooled = attention.len() * ch
#let layers = (shape: (pooled,model.hidden,model.hidden,1), invariant: (pooled,model.hidden,1))
