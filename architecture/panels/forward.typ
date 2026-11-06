#import "../config.typ": *
#import "../drawing.typ": *

// Lengths are in the original figure's drawing units. Internal y offsets are
// measured from the stage bottom. Every stage receives its own left edge and width.
#let layout = (left: 3.25, bottom: 55, height: 31, flow: 11, caption: -7.8,
  caption-gap: 3.1, corner: 1.7, columns: 2)
#let inputs = (size: 10.5, rows: (19,3.5), label-rise: 7.6)
#let orbit-grid = (bottom: -2.5, height: 37, radius: 2.1, first: 22.8,
  pitch: (9.6,6.2), cell: (8.5,4.7), radius-cell: 0.7, title: 31.9, note: 0.6)
#let cnn = (inset: 2, width: 8, height: 6.5, rows: (15.2,4.4), fusion: (8.5,10),
  fusion-y: 8, ports: (6.5,3.3), bend: 8deg, title: 29.4, note: 24.3)
#let sheets = (large: (4.6,12.5,5,(0.75,0.65)), small: (4,5.5,2,(0.55,0.45)),
  tilt: 0.8, transparency: 85%, step: 5%, feature-y: 7, label-y: 24.2)
#let means = (inset: 3.2, tile-y: 20.6, tile-h: 6.5, rows: (15,8.8), note: 3.2, compact: 86%)
#let maps = (top: 19, pitch: 8, label-x: 5.8, rows: (([$Psi_1$],"head"),([$Psi_2$],"head"),([scalar],"feature")))
#let heat = (title: 28, range-y: 23.8, top: 18.4, grid-height: 13.9, pitch: (7.7,7.8), size: 6.1, note: 3.5)
#let output = (inset: 2.3, height: 8, rows: (17.8,4.5), label-gap: 0.7)

#let y(offset) = layout.bottom + offset
#let outer(x,w,role) = frame(x,y(0),w,layout.height,role,r: layout.corner,stroke: pen.frame)
#let grid-cell(i) = (calc.rem(i,layout.columns),calc.quo(i,layout.columns))
#let input-stage(x,w) = {
  for (row,role,title) in inputs.rows.zip(("galaxy","psf"),([galaxy],[PSF])) {
    picture(x+w/2,y(row),inputs.size,role+".png",role)
    label(x+w/2,y(row+inputs.label-rise),title,size: fs.body,role: role,bold: true)
  }
}
#let orbit-stage(x,w) = {
  let g = orbit-grid
  frame(x,y(g.bottom),w,g.height,"orbit",fill: fills.orbit,r: g.radius,stroke: pen.frame)
  label(x+w/2,y(g.title),[Joint $D_4$ orbit],size: fs.title,bold: true)
  let (cw,h) = g.cell
  let row-width = cw + (layout.columns - 1)*g.pitch.first()
  for (i,transform) in orbit.enumerate() {
    let (col,row) = grid-cell(i)
    tile(x+(w - row-width)/2+col*g.pitch.first(),y(g.first - row*g.pitch.last()),cw,h,"orbit",transform.op,
      fill: white,r: g.radius-cell,stroke: pen.inner)
  }
  label(x+w/2,y(g.note),[galaxy and PSF \ transformed jointly],size: fs.tiny,role: "muted",italic: true)
}
#let fusion-stage(x,w) = {
  outer(x,w,"fusion")
  label(x+w/2,y(cnn.title),[$F$],size: fs.group,role: "fusion")
  label(x+w/2,y(cnn.note),[one parameter set \ for all #orbit.len() variants],size: fs.tiny,role: "muted",italic: true)
  let (fw,fh) = cnn.fusion
  let fx = x+w - cnn.inset - fw
  tile(fx,y(cnn.fusion-y),fw,fh,"fusion",[fusion],size: fs.body,bold: true)
  for (row,role,name,port,bend) in cnn.rows.zip(("galaxy","psf"),([galaxy],[PSF]),cnn.ports,(cnn.bend,-cnn.bend)) {
    tile(x+cnn.inset,y(row),cnn.width,cnn.height,role,[#name \ CNN],bold: true)
    wire(x+cnn.inset+cnn.width,y(row+cnn.height/2),fx,y(cnn.fusion-y+port),role: role,bend: bend)
  }
}
#let stack-maps(x,bottom,spec,role) = {
  let (w,h,count,(dx,dy)) = spec
  let tilt = sheets.tilt
  let c = colors.at(role)
  for k in range(count).rev() {
    let sheet = polygon((0pt,h*uy),(w*ux,h*uy),((w+tilt)*ux,0pt),(tilt*ux,0pt),
      stroke: pen.sheet+c, fill: c.transparentize(sheets.transparency - (count - k)*sheets.step))
    f.node(pt(x+k*dx+(w+tilt)/2,bottom+k*dy+h/2),sheet,layer: z.sheet)
  }
}
#let features-stage(x,w) = {
  stack-maps(x,y(sheets.feature-y),sheets.large,"feature")
  label(x+w/2,y(sheets.label-y),[$phi_0, dots.h, phi_(#(orbit.len() - 1))$],size: fs.body,role: "feature")
}
#let reynolds-stage(x,w) = {
  outer(x,w,"orbit")
  let aligned = $psi_i^(-1) phi_i$
  tile(x+means.inset,y(means.tile-y),w - 2*means.inset,means.tile-h,"orbit",[inverse-align \ #aligned],size: fs.tiny)
  let equations = ([$Psi_(1,2) = 1/#orbit.len() limits(sum)_i w_i #aligned$],[$1/#orbit.len() limits(sum)_i #aligned$])
  for (row,body) in means.rows.zip(equations) {
    label(x+w/2,y(row),scale(x: means.compact,reflow: true,body),size: fs.equation)
  }
  label(x+w/2,y(means.note),[signed / sign-free \ averages],size: fs.tiny,role: "muted")
}
#let map-stage(x,w) = {
  for (i,(name,role)) in maps.rows.enumerate() {
    let row = y(maps.top - i*maps.pitch)
    stack-maps(x,row,sheets.small,role)
    label(x+maps.label-x,row+sheets.small.at(1)/2,name,anchor: left,size: if role == "head" {fs.body} else {fs.layer})
  }
}
#let pooling-stage(x,w) = {
  outer(x,w,"pool")
  let cx = x+w/2
  label(cx,y(heat.title),[Attention maps],role: "pool",bold: true)
  label(cx,y(heat.range-y),[maps 1–#attention.len()],size: fs.tiny,role: "muted")
  let rows = calc.ceil(attention.len()/layout.columns)
  let pitch = calc.min(heat.pitch.last(),heat.grid-height/rows)
  let size = calc.min(heat.size,pitch)
  for (i,asset) in attention.enumerate() {
    let (col,row) = grid-cell(i)
    picture(cx+(col - (layout.columns - 1)/2)*heat.pitch.first(),y(heat.top - row*pitch),size,asset,"pool")
  }
  label(cx,y(heat.note),[from scalar map])
}
#let heads-stage(x,w) = {
  outer(x,w,"head")
  for (row,role,body,out,size) in output.rows.zip(("head","feature"),([bias-free \ odd MLP],[invariant \ MLP]),
      ([$e'_1,e'_2$],[size, flux]),(fs.group,fs.title)) {
    tile(x+output.inset,y(row),w - 2*output.inset,output.height,role,body,bold: true)
    let cy = y(row+output.height/2)
    wire(x+w,cy,x+w+gap-x,cy,role: role)
    label(x+w+gap-x+output.label-gap,cy,out,role: role,size: size,bold: true,anchor: left)
  }
}

// Stage definitions. The flow in config.typ controls which ones appear and in what order.
#let stage(draw,width,title,sub) = (
  draw: draw, width: width, title: title, sub: sub)
#let pipeline = (
  inputs: stage(input-stage,inputs.size,[Inputs],[$#model.pixels times #model.pixels$ pixels]),
  orbit: stage(orbit-stage,23,[Joint orbit],[#orbit.len() joint variants]),
  fusion: stage(fusion-stage,24.5,[Shared $F$],[shared weights]),
  features: stage(features-stage,8.5,[Features $phi_i$],[$#orbit.len() times #model.side^2 times #ch$]),
  reynolds: stage(reynolds-stage,23.5,[Reynolds],[signed / invariant]),
  maps: stage(map-stage,10,[$Psi$ maps],[$#maps.rows.len() times #model.side^2 times #ch$]),
  pooling: stage(pooling-stage,21.5,[Pooling],[#attention.len() spatial maps]),
  heads: stage(heads-stage,20.5,[Output heads],[shape + scalars]),
)
#let draw() = {
  let x = layout.left
  for (i,id) in flow.keys().enumerate() {
    let s = pipeline.at(id)
    let number = step(id)
    (s.draw)(x,s.width)
    let mid = x+s.width/2
    let title = if number == none {s.title} else {[#number. #s.title]}
    label(mid,y(layout.caption),title,bold: number != none,italic: number == none)
    label(mid,y(layout.caption - layout.caption-gap),s.sub,size: fs.tiny,role: "muted")
    x += s.width
    if i + 1 < flow.len() {join(x,x+gap-x,y(layout.flow),"line")}
    x += gap-x
  }
}
