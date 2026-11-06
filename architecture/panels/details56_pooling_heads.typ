#import "../config.typ": *
#import "../drawing.typ": *

#let layout = (left: 125, bottom: 2, width: 53, height: 38.5, margin: 3, title-inset: 2,
  title-y: 37.5, strip-y: 31.3, strip-radius: 0.5, frame-radius: 1.2,
  sums-y: 24.6, features-y: 21.2, fork-y: 20.1, footer-y: 4.2)
#let head = (bottom: 7, height: 11, gap: 1.6, padding: 1.4, title-offset: 9,
  layer-offset: 4.3, layer-height: 2.8, output-offset: 1.8, radius: 0.65,
  layer-radius: 0.35, layer-width: 3.2, last-width: 2.3, scalar-width: 3.4, bend: 6deg)
#let operations = (
  (w: 5.5,h: 4.4,role: "feature",body: [scalar],size: fs.small,fill: fills.feature),
  (w: 9.2,h: 5,role: "pool",body: [Dense #ch \ GELU],size: fs.layer,fill: fills.pool),
  (w: 9.6,h: 5,role: "pool",body: [Dense #attention.len() \ softmax$attach("",b:p)$],size: fs.head,fill: fills.pool),
  (w: 12.2,h: 4.4,role: "pool",body: [maps 1–#attention.len()],size: fs.layer,fill: white),
)
#let heads = (
  (role: "head",title: [odd shape heads],layers: layers.shape, width: head.layer-width,
    last: head.last-width,output: [tanh $arrow.r e'_1,e'_2$]),
  (role: "feature",title: [invariant heads],layers: layers.invariant,width: head.scalar-width,
    last: head.scalar-width,output: [GELU $arrow.r$ size, flux]),
)
#let layer-widths(s) = s.layers.enumerate().map(((i,n)) => if i+1 == s.layers.len() {s.last} else {s.width})
#let draw-head(x,w,s) = {
  let cx = x+w/2
  frame(x,head.bottom,w,head.height,s.role,fill: fills.at(s.role),r: head.radius,stroke: pen.head)
  label(cx,head.bottom+head.title-offset,s.title,role: s.role,bold: true,size: fs.tiny)
  let widths = layer-widths(s)
  let start = cx - span(widths,gap-x)/2
  let y = head.bottom+head.layer-offset
  for (i,n) in s.layers.enumerate() {
    let width = widths.at(i)
    tile(start,y,width,head.layer-height,s.role,str(n),fill: white,r: head.layer-radius,stroke: pen.fine,size: fs.head)
    start += width
    if i+1 < widths.len() {join(start,start+gap-x,y+head.layer-height/2,s.role)}
    start += gap-x
  }
  label(cx,head.bottom+head.output-offset,s.output,size: fs.tiny)
}
#let draw() = {
  let g = layout
  let strip-width = span(operations.map(op => op.w),gap-x)
  let min-head = heads.map(s => span(layer-widths(s),gap-x)+2*head.padding)
  let heads-width = span((calc.max(..min-head),)*heads.len(),head.gap)
  let width = calc.max(g.width,calc.max(strip-width,heads-width)+2*g.margin)
  let cx = g.left+width/2
  frame(g.left,g.bottom,width,g.height,"pool",stroke: pen.rim,r: g.frame-radius)
  label(g.left+g.title-inset,g.title-y,detail-title(("pooling","heads"),[pooling + heads]),size: fs.title,bold: true,anchor: left)
  let x = cx - strip-width/2
  for (i,op) in operations.enumerate() {
    tile(x,g.strip-y - op.h/2,op.w,op.h,op.role,op.body,fill: op.fill,r: g.strip-radius,size: op.size)
    x += op.w
    if i+1 < operations.len() {join(x,x+gap-x,g.strip-y,"pool")}
    x += gap-x
  }
  label(cx,g.sums-y,[Spatially weighted sums of shape maps],size: fs.formula)
  let count = if attention.len() == 4 {[Four]} else {str(attention.len())}
  label(cx,g.features-y,[#count $#ch$-channel vectors $arrow.r #pooled$ features])
  let head-width = (width - 2*g.margin - (heads.len() - 1)*head.gap)/heads.len()
  for (i,s) in heads.enumerate() {
    let left = g.left+g.margin+i*(head-width+head.gap)
    let mid = left+head-width/2
    draw-head(left,head-width,s)
    wire(cx,g.fork-y,mid,head.bottom+head.height,role: s.role,
      bend: if mid < cx {-head.bend} else {head.bend})
  }
  label(cx,g.footer-y,[Shared pooling weights preserve $D_4$ symmetry.],role: "muted",size: fs.tiny)
}
