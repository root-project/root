## \file
## \ingroup tutorial_v7
##
## This ROOT 7 example demonstrates how to create a ROOT 7 canvas (RCanvas) and
## draw ROOT 7 boxes in it (RBox). It generates a set of boxes using the
## "normal" coordinates' system.
## Run macro with python3 -i box.py command to get interactive canvas
##
## \macro_image (rcanvas_js)
## \macro_code
##
## \date 2021-06-15
## \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
## is welcome!
## \author Sergey Linev

import ROOT
from ROOT.Experimental import RCanvas, RBox, RPadPos, RColor

# Create a canvas to be displayed.
canvas = RCanvas.Create("RBox drawing")

box1 = canvas.Draw[RBox](RPadPos(0.1, 0.1), RPadPos(0.3, 0.6))
box1.border.color.Set(RColor.kBlue)
box1.border.width.Set(5)
box1.fill.color = RColor(0, 255, 0, 127) # 50% opaque

box2 = canvas.Draw[RBox](RPadPos(0.4, 0.2), RPadPos(0.6, 0.7))
box2.border.color = RColor.kRed
box2.border.width = 10
box2.border.style = 2
box2.fill.color = RColor(0, 0, 255, 179) # 70% opaque

box3 = canvas.Draw[RBox](RPadPos(0.7, 0.4), RPadPos(0.9, 0.6))
box3.border.width = 3
box3.fill.color = RColor.kBlue

box4 = canvas.Draw[RBox](RPadPos(0.7, 0.7), RPadPos(0.9, 0.9))
box4.border.width = 4

box5 = canvas.Draw[RBox](RPadPos(0.7, 0.1), RPadPos(0.9, 0.3))
box5.border.rx = 10
box5.border.ry = 10
box5.border.width = 2

canvas.Show()
