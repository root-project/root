## \file
## \ingroup tutorial_roostats
## \notebook -js
## High Level Factory: creation of a simple model
##
## \macro_image
## \macro_output
## \macro_code
##
## \date July 2022
## \authors Artem Busorgin, Danilo Piparo (C++ version)

import ROOT

# --- Build the datacard and dump to file---
card_name = "HLFavtoryexample.rs"
with open(card_name, "w") as f:
    f.write("// The simplest card\n\n")
    f.write("gauss = Gaussian(mes[5.20,5.30],mean[5.28,5.2,5.3],width[0.0027,0.001,1]);\n")
    f.write("argus = ArgusBG(mes,5.291,argpar[-20,-100,-1]);\n")
    f.write("sum = SUM(nsig[200,0,10000]*gauss,nbkg[800,0,10000]*argus);\n\n")

hlf = ROOT.RooStats.HLFactory("HLFavtoryexample", card_name, False)

# --- Take elements out of the internal workspace ---
w = hlf.GetWs()

mes = w.arg("mes")
sum = w.pdf("sum")
argus = w.pdf("argus")

# --- Generate a toyMC sample from composite PDF ---
data = sum.generate(mes, 2000)

# --- Perform extended ML fit of composite PDF to toy data ---
sum.fitTo(data)

# --- Plot toy data and composite PDF overlaid ---
mesframe = mes.frame()
data.plotOn(mesframe)
sum.plotOn(mesframe)
sum.plotOn(mesframe, Components=argus, LineStyle=ROOT.kDashed)

ROOT.gROOT.SetStyle("Plain")

c = ROOT.TCanvas()
mesframe.Draw()

c.SaveAs("rs601_HLFactoryexample.png")
