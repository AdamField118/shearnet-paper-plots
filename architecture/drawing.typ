#import "@preview/fletcher:0.5.8" as f: diagram, node, edge

#import "config.typ": *
#let panel = sys.inputs.at("panel", default: "forward")
#assert(panel in views, message: "Unknown architecture panel: " + panel)
#let view = views.at(panel)
#let (ux,uy) = view.size.zip(view.extent).map(((size, extent)) => size / extent)
#let pt(x,y) = (x*ux, y*uy)
// Fletcher uses numeric paint order; name the layers so new marks stack consistently.
#let z = (viewport: -1, box: 0, sheet: 1, image: 2, label: 3, edge: 4)
#let frame(x,y,w,h,role, fill: white, r: 0.8, stroke: pen.box, opacity: 100%) = node(
  pt(x+w/2,y+h/2), width: w*ux, height: h*uy, shape: f.shapes.rect,
  stroke: stroke + colors.at(role).transparentize(100% - opacity), fill: fill, corner-radius: r*ux, layer: z.box)
#let label(x,y,body,size: fs.small,role: "ink",bold: false,italic: false,anchor: center) = {
  let content = text(size: size, fill: colors.at(role), weight: if bold {"bold"} else {"regular"},
    style: if italic {"italic"} else {"normal"}, align(center, body))
  let dx = if anchor == center {0pt} else {measure(content).width/2 * (if anchor == left {1} else {-1})}
  node((x*ux+dx,y*uy), content, layer: z.label)
}
#let tile(x,y,w,h,role,body, fill: auto, r: 0.8, stroke: pen.box, opacity: 100%, ..font) = {
  frame(x,y,w,h,role, fill: if fill == auto {fills.at(role)} else {fill}, r: r, stroke: stroke, opacity: opacity)
  label(x+w/2,y+h/2,body,..font)
}
#let wire(x,y,X,Y,role: "line",width: pen.arrow,bend: 0deg,tip: true,opacity: 100%) = {
  let a = pt(x,y)
  let b = pt(X,Y)
  let dx = b.first() - a.first()
  let dy = b.last() - a.last()
  let distance = calc.sqrt(calc.pow(dx/1pt,2) + calc.pow(dy/1pt,2))
  let trim = if tip {pen.clearance/1pt / distance} else {0}
  edge((a.first()+dx*trim,a.last()+dy*trim), (b.first()-dx*trim,b.last()-dy*trim),
    if tip {"-|>"} else {"-"}, stroke: width + colors.at(role).transparentize(100% - opacity), bend: bend,
    mark-scale: pen.tip-scale * (pen.tip-reference / width), snap-to: (none, none), layer: z.edge)
}
#let picture(x,y,size,asset,role) = {
  node(pt(x,y), image("stamps/" + asset, width: size*ux, height: size*uy, fit: "stretch"), layer: z.image)
  node(pt(x,y), width: size*ux, height: size*uy, shape: f.shapes.rect,
    stroke: (if role == "pool" {pen.fine} else {pen.rim}) + colors.at(role), layer: z.label)
}
#let render(body) = diagram(node-inset: 0pt,node-outset: 0pt,node-stroke: none,node-shape: f.shapes.rect,
  node((view.origin.first()*ux+view.size.first()/2,view.origin.last()*uy+view.size.last()/2),
    width: view.size.first(),height: view.size.last(),layer: z.viewport),
  body)

// Join adjacent boxes on a horizontal row, leaving the same clearance at each end.
#let join(right, left, y, role, width: pen.frame) = wire(
  right + pen.inset, y, left - pen.inset, y, role: role, width: width)
// Sum of the occupied width of a row, including its internal gaps.
#let span(widths, gap) = widths.sum(default: 0) + calc.max(0, widths.len() - 1) * gap
