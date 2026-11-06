#import "../config.typ": *
#import "../drawing.typ": *

#let layout = (left: 31, bottom: 2, height: 38.5, margin: 2.2, title-y: 37.5,
  footer-y: 4.8, label-gutter: 11, galaxy-y: 24.9, psf-y: 13.1, row-height: 7.4,
  stem-width: 7, residual-width: 6.8, downsample-width: 8.5,
  radius: 0.55, frame-radius: 1.2, dilation-width: 9.5, dilation-height: 8)
#let fusion = (gap: 4.3, bottom: 7.3, width: 21.9, height: 27, inset: 1.4,
  title-offset: 2.3, first-offset: 5.2, footer-offset: 2.9, footer-space: 6.7, radius: 0.4,
  galaxy-port: 19.9, psf-port: 11.9, galaxy-bend: 8deg, psf-bend: -6deg)
#let fusion-layers = (
  (height: 3.6, body: [$1 times 1$ projection \ + positions]),
  (height: 4, body: [cross-attn \ G:Q; PSF:K,V]),
  (height: 3.4, body: [self-attn + FFN]),
)
#let widths = (layout.stem-width,) + stages.map(s => if s.downsample {layout.downsample-width} else {layout.residual-width})
#let encoder(role,name,row,start) = {
  let cy = row+layout.row-height/2
  label(layout.left+layout.margin,cy,name,role: role,anchor: left,bold: true)
  let x = start
  for (i,w) in widths.enumerate() {
    let body = if i == 0 {[stem \ #(model.pixels)²×#stages.first().channels]} else {
      let s = stages.at(i - 1)
      if s.downsample {[$arrow.b #model.stride$ \ ]}
      [Res ×#s.at(role) \ #s.channels ch]
    }
    tile(x,row,w,layout.row-height,role,body,size: if i == 0 {fs.narrow} else {fs.layer},r: layout.radius,stroke: pen.inner)
    x += w
    if i + 1 < widths.len() {join(x,x+gap-x,cy,role)}
    x += gap-x
  }
}
#let draw() = {
  let start = layout.left+layout.label-gutter
  let encoder-end = start+span(widths,gap-x)
  let dilation-x = encoder-end+gap-x
  let fusion-x = dilation-x+layout.dilation-width+fusion.gap
  let gaps = calc.max(0,fusion-layers.len() - 1)*gap-y
  let stack-height = fusion-layers.map(l => l.height).sum(default: 0)+gaps
  // Footer space includes its two lines of text beneath the last layer.
  let fusion-height = calc.max(fusion.height,fusion.first-offset+stack-height+fusion.footer-space)
  let extra = fusion-height - fusion.height
  let right = fusion-x+fusion.width+layout.margin
  frame(layout.left,layout.bottom,right - layout.left,layout.height+extra,"fusion",stroke: pen.detail,r: layout.frame-radius)
  label(layout.left+layout.margin,layout.title-y+extra,detail-title(("fusion",),[shared per-variant $F$]),anchor: left,size: fs.title,bold: true)
  label(right - layout.margin,layout.title-y+extra,[distinct galaxy / PSF weights],anchor: right,size: fs.tiny,role: "muted",italic: true)
  encoder("galaxy",[Galaxy],layout.galaxy-y,start)
  encoder("psf",[PSF],layout.psf-y,start)
  let gy = layout.galaxy-y+layout.row-height/2
  let py = layout.psf-y+layout.row-height/2
  let dilations = model.dilations.map(n => str(n)).join(",")
  tile(dilation-x,gy - layout.dilation-height/2,layout.dilation-width,layout.dilation-height,"galaxy",
    [dilated \ $#model.kernel times #model.kernel$ \ $d=#dilations$],size: fs.detail,r: layout.radius,stroke: pen.inner)
  join(encoder-end,dilation-x,gy,"galaxy")
  frame(fusion-x,fusion.bottom,fusion.width,fusion-height,"fusion",fill: fills.fusion,stroke: pen.rim)
  let cx = fusion-x+fusion.width/2
  let top = fusion.bottom+fusion-height
  label(cx,top - fusion.title-offset,[Transformer \ fusion],size: fs.tiny,role: "fusion",bold: true)
  let layer-top = top - fusion.first-offset
  for (i,layer) in fusion-layers.enumerate() {
    let bottom = layer-top - layer.height
    tile(fusion-x+fusion.inset,bottom,fusion.width - 2*fusion.inset,layer.height,"fusion",layer.body,
      fill: white,size: fs.narrow,r: fusion.radius,stroke: pen.fusion)
    layer-top = bottom - gap-y
    if i + 1 < fusion-layers.len() {
      wire(cx,bottom,cx,layer-top,role: "fusion")
    }
  }
  label(cx,fusion.bottom+fusion.footer-offset,[$d=#ch$; $h=#model.heads$ \ FFN width #model.ffn],size: fs.detail,role: "muted")
  wire(dilation-x+layout.dilation-width,gy,fusion-x,fusion.bottom+fusion.galaxy-port,role: "galaxy",bend: fusion.galaxy-bend)
  wire(encoder-end,py,fusion-x,fusion.bottom+fusion.psf-port,role: "psf",bend: fusion.psf-bend)
  label((layout.left+right)/2,layout.footer-y,[LayerNorm + GELU; anti-aliased downsampling; outputs $#model.side^2 times #ch$.],size: fs.tiny,role: "muted")
}
