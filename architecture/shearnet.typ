#import "config.typ": typeface, fs, colors, pen
#import "drawing.typ": panel, render
#set page(width: auto, height: auto, margin: pen.pad)
#set text(font: typeface.text, size: fs.small, fill: colors.ink)
#show math.equation: set text(font: typeface.math)
#set par(leading: pen.leading)
#import ("panels/" + panel + ".typ") as selected
#context render(selected.draw())
