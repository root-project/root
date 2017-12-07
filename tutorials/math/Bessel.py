## \file
## \ingroup tutorial_math
## \notebook
## Show the different kinds of Bessel functions available in ROOT
## To execute the macro type in:
##
## ~~~{.cpp}
## root[0] .x Bessel.C
## ~~~
##
## It will create one canvas with the representation
## of the  cylindrical and spherical Bessel functions
## regular and modified
##
## Based on Bessel.C by Magdalena Slawinska
##
## \macro_image
## \macro_code
##
## \author Juan Fernando Jaramillo Botero

from ROOT import TCanvas, TF1, gSystem, gPad, TLegend, TPaveLabel, kBlack


gSystem.Load("libMathMore")

DistCanvas = TCanvas("DistCanvas", "Bessel functions example", 10, 10, 800, 600)
DistCanvas.SetFillColor(17)
DistCanvas.Divide(2, 2)
DistCanvas.cd(1)
gPad.SetGrid()
gPad.SetFrameFillColor(19)
leg = TLegend(0.75, 0.7, 0.89, 0.89)

n = 5 # number of functions in each pad
# drawing the set of Bessel J functions
JBessel = []
for nu in range(n):
   _jbessel = TF1("J_0", "ROOT::Math::cyl_bessel_j([0],x)", 0, 10)
   _jbessel.SetParameters(nu, 0.0)
   _jbessel.SetTitle("")
   _jbessel.SetLineStyle(1)
   _jbessel.SetLineWidth(3)
   _jbessel.SetLineColor(nu + 1)
   JBessel.append(_jbessel)
#No se porque esta linea estaba asi: JBessel[0].TF1.GetXaxis().SetTitle("x")
JBessel[0].GetXaxis().SetTitle("x")
JBessel[0].GetXaxis().SetTitleSize(0.06)
JBessel[0].GetXaxis().SetTitleOffset(.7)

# setting the title in a label style
p1 = TPaveLabel(.0, .90, .0 + .50, .90 + .10, "Bessel J functions", "NDC")
p1.SetFillColor(0)
p1.SetTextFont(22)
p1.SetTextColor(kBlack)

# setting the legend
leg.AddEntry(JBessel[0].DrawCopy(), " J_0(x)", "l")
leg.AddEntry(JBessel[1].DrawCopy("same"), " J_1(x)", "l")
leg.AddEntry(JBessel[2].DrawCopy("same"), " J_2(x)", "l")
leg.AddEntry(JBessel[3].DrawCopy("same"), " J_3(x)", "l")
leg.AddEntry(JBessel[4].DrawCopy("same"), " J_4(x)", "l")

leg.Draw()
p1.Draw()

# ------------------------------------------------
DistCanvas.cd(2)
gPad.SetGrid()
gPad.SetFrameFillColor(19)

leg2 = TLegend(0.75, 0.7, 0.89, 0.89)
# ------------------------------------------------
# Drawing Bessel k
KBessel = []
for nu in range(n):
  _kbessel = TF1("J_0", "ROOT::Math::cyl_bessel_k([0],x)", 0, 10)
  _kbessel.SetParameters(nu, 0.0)
  _kbessel.SetTitle("Bessel K functions")
  _kbessel.SetLineStyle(1)
  _kbessel.SetLineWidth(3)
  _kbessel.SetLineColor(nu+1)
  KBessel.append(_kbessel)
KBessel[0].GetXaxis().SetTitle("x")
KBessel[0].GetXaxis().SetTitleSize(0.06)
KBessel[0].GetXaxis().SetTitleOffset(.7)

# setting title
p2 = TPaveLabel(.0, .90, .0 + .50, .90 + .10, "Bessel K functions", "NDC")
p2.SetFillColor(0)
p2.SetTextFont(22)
p2.SetTextColor(kBlack)

# setting legend
leg2.AddEntry(KBessel[0].DrawCopy(), " K_0(x)", "l")
leg2.AddEntry(KBessel[1].DrawCopy("same"), " K_1(x)", "l")
leg2.AddEntry(KBessel[2].DrawCopy("same"), " K_2(x)", "l")
leg2.AddEntry(KBessel[3].DrawCopy("same"), " K_3(x)", "l")
leg2.AddEntry(KBessel[4].DrawCopy("same"), " K_4(x)", "l")
leg2.Draw()
p2.Draw()
# ------------------------------------------------
DistCanvas.cd(3)
gPad.SetGrid()
gPad.SetFrameFillColor(19)
leg3 = TLegend(0.75, 0.7, 0.89, 0.89)
# ------------------------------------------------
# Drawing Bessel i
iBessel = []
for nu in range(5):
  _ibessel= TF1("J_0", "ROOT::Math::cyl_bessel_i([0],x)", 0, 10)
  _ibessel.SetParameters(nu, 0.0)
  _ibessel.SetTitle("Bessel I functions")
  _ibessel.SetLineStyle(1)
  _ibessel.SetLineWidth(3)
  _ibessel.SetLineColor(nu + 1)
  iBessel.append(_ibessel)

iBessel[0].GetXaxis().SetTitle("x")
iBessel[0].GetXaxis().SetTitleSize(0.06)
iBessel[0].GetXaxis().SetTitleOffset(.7)

# setting title
p3 = TPaveLabel(.0, .90, .0 + .50, .90 + .10, "Bessel I functions", "NDC")
p3.SetFillColor(0)
p3.SetTextFont(22)
p3.SetTextColor(kBlack)

# setting legend
leg3.AddEntry(iBessel[0].DrawCopy(), " I_0", "l")
leg3.AddEntry(iBessel[1].DrawCopy("same"), " I_1(x)", "l")
leg3.AddEntry(iBessel[2].DrawCopy("same"), " I_2(x)", "l")
leg3.AddEntry(iBessel[3].DrawCopy("same"), " I_3(x)", "l")
leg3.AddEntry(iBessel[4].DrawCopy("same"), " I_4(x)", "l")
leg3.Draw()
p3.Draw()
# ------------------------------------------------
DistCanvas.cd(4)
gPad.SetGrid()
gPad.SetFrameFillColor(19)
leg4 = TLegend(0.75, 0.7, 0.89, 0.89)
# ------------------------------------------------
# Drawing sph_bessel
jBessel = []
for nu in range(5):
  _jbessel = TF1("J_0", "ROOT::Math::sph_bessel([0],x)", 0, 10)
  _jbessel.SetParameters(nu, 0.0)
  _jbessel.SetTitle("Bessel j functions")
  _jbessel.SetLineStyle(1)
  _jbessel.SetLineWidth(3)
  _jbessel.SetLineColor(nu+1)
  jBessel.append(_jbessel)
jBessel[0].GetXaxis().SetTitle("x")
jBessel[0].GetXaxis().SetTitleSize(0.06)
jBessel[0].GetXaxis().SetTitleOffset(.7)

# setting title
p4 = TPaveLabel(.0, .90 , .0 + .50, .90 + .10, "Bessel j functions", "NDC")
p4.SetFillColor(0)
p4.SetTextFont(22)
p4.SetTextColor(kBlack)

# setting legend

leg4.AddEntry(jBessel[0].DrawCopy(), " j_0(x)", "l")
leg4.AddEntry(jBessel[1].DrawCopy("same"), " j_1(x)", "l")
leg4.AddEntry(jBessel[2].DrawCopy("same"), " j_2(x)", "l")
leg4.AddEntry(jBessel[3].DrawCopy("same"), " j_3(x)", "l")
leg4.AddEntry(jBessel[4].DrawCopy("same"), " j_4(x)", "l")

leg4.Draw()
p4.Draw()

DistCanvas.cd()
