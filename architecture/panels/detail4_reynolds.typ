#import "../config.typ": *
#import "../drawing.typ": *

#let layout = (left: 2, bottom: 2, width: 76, height: 26, radius: 1.2,
  title-inset: (2,2.6), label-inset: 3, grid-inset: 7, row-y: (18.1,12.1),
  pair-width: 4.7, pair-height: 3.3, pair-gap: 0.9, pair-radius: 0.38,
  summary-x: 43, text-y: 16.8, divider-x: 56.2, divider-y: (8,22), response-x: 67)
#let draw() = {
  let g = layout
  frame(g.left,g.bottom,g.width,g.height,"orbit",fill: fills.orbit,r: g.radius,stroke: pen.rim)
  label(g.left+g.title-inset.first(),g.bottom+g.height - g.title-inset.last(),
    detail-title(("reynolds",),[Reynolds weights]),anchor: left,size: fs.title,bold: true)
  for (i,key) in ("w1","w2").enumerate() {
    let signs = orbit.map(transform => transform.at(key))
    let y = g.row-y.at(i)
    label(g.left+g.label-inset,y+g.pair-height/2,[$w_(#(i+1))$],anchor: left)
    for j in range(calc.ceil(signs.len()/2)) {
      let pair = signs.slice(2*j,calc.min(2*j+2,signs.len())).join("  ")
      tile(g.left+g.grid-inset+j*(g.pair-width+g.pair-gap),y,g.pair-width,g.pair-height,"orbit",pair,
        fill: white.transparentize(38%),r: g.pair-radius,opacity: 70%,stroke: pen.fine,size: fs.tiny)
    }
  }
  label(g.summary-x,g.text-y,[signed mean $arrow.r Psi_1,Psi_2$ \ ordinary mean $arrow.r$ scalar],role: "muted")
  f.edge(pt(g.divider-x,g.divider-y.first()),pt(g.divider-x,g.divider-y.last()),"-",
    stroke: pen.inner + colors.orbit.transparentize(45%),snap-to: (none,none),layer: z.edge)
  label(g.response-x,g.text-y,[Response term \ $K=#model.psf-orbits$ PSF-only orbits])
}
