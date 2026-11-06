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
// Panel geometry uses drawing units. Convert the common edge-to-edge span
// separately on each axis so vertical and horizontal links match on paper.
#let gap-x = connector.span / ux
#let gap-y = connector.span / uy
#let wire(x,y,X,Y,role: "line",bend: 0deg) = {
  let (ax,ay) = pt(x,y)
  let (bx,by) = pt(X,Y)
  let dx = bx - ax
  let dy = by - ay
  let length = calc.sqrt(calc.pow(dx/1pt,2) + calc.pow(dy/1pt,2))*1pt
  let trim = connector.clearance / length
  edge((ax+dx*trim,ay+dy*trim), (bx - dx*trim,by - dy*trim),connector.head,
    stroke: connector.stroke + colors.at(role),bend: bend,mark-scale: connector.head-scale,
    snap-to: (none,none),layer: z.edge)
}
#let picture(x,y,size,asset,role,scaling: auto) = {
  node(pt(x,y), image("stamps/" + asset, width: size*ux, height: size*uy, fit: "stretch", scaling: scaling), layer: z.image)
  node(pt(x,y), width: size*ux, height: size*uy, shape: f.shapes.rect,
    stroke: (if role == "pool" {pen.fine} else {pen.rim}) + colors.at(role), layer: z.label)
}
#let render(body) = diagram(node-inset: 0pt,node-outset: 0pt,node-stroke: none,node-shape: f.shapes.rect,
  node((view.origin.first()*ux+view.size.first()/2,view.origin.last()*uy+view.size.last()/2),
    width: view.size.first(),height: view.size.last(),layer: z.viewport),
  body)

#let join(right,left,y,role) = wire(right,y,left,y,role: role)
// Sum of the occupied width of a row, including its internal gaps.
#let span(widths, gap) = widths.sum(default: 0) + calc.max(0, widths.len() - 1) * gap
