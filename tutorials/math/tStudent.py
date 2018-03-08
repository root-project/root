## \file
## \ingroup tutorial_math
## \notebook
## Example macro describing the student t distribution
##
## ~~~{.cpp}
## root[0]: .x tStudent.C
## ~~~
##
## It draws the pdf, the cdf and then 10 quantiles of the t Student distribution
##
## based on Magdalena Slawinska's tStudent.C
##
## \macro_image
## \macro_code
##
## \author Juan Fernando Jaramillo Botero

from ROOT import TH1D, TF1, TCanvas, kRed, kBlue
import ROOT


# this is the way to force load of MathMore in Cling
ROOT.Math.MathMoreLibrary.Load()

n = 100
a = -5.
b = 5.
pdf = TF1("pdf", "ROOT::Math::tdistribution_pdf(x,3.0)", a, b)
cum = TF1("cum", "ROOT::Math::tdistribution_cdf(x,3.0)", a, b)

quant = TH1D("quant", "", 9, 0, 0.9)

for i in range(1, 10):
    quant.Fill((i-0.5)/10.0, ROOT.Math.tdistribution_quantile((1.0 * i) / 10,
                                                              3.0))

xx = []
xx.append(-1.5)
for i in range(1, 9):
    xx.append(quant.GetBinContent(i))
xx.append(1.5)
pdfq = []
for i in range(9):
    nbin = int(n * (xx[i+1] - xx[i]) / 3.0 + 1.0)
    name = "pdf"
    name += str(i)
    pdfq.append(TH1D(name, "", nbin, xx[i], xx[i+1]))
    for j in range(1, nbin):
        x = j * (xx[i+1] - xx[i]) / nbin + xx[i]
        pdfq[i].SetBinContent(j, ROOT.Math.tdistribution_pdf(x, 3))

Canvas = TCanvas("DistCanvas", "Student Distribution graphs", 10, 10, 800, 700)
pdf.SetTitle("Student t distribution function")
cum.SetTitle("Cumulative for Student t")
quant.SetTitle("10-quantiles  for Student t")
Canvas.Divide(2, 2)
Canvas.cd(1)
pdf.SetLineWidth(2)
pdf.DrawCopy()
Canvas.cd(2)
cum.SetLineWidth(2)
cum.SetLineColor(kRed)
cum.Draw()
Canvas.cd(3)
quant.Draw()
quant.SetLineWidth(2)
quant.SetLineColor(kBlue)
quant.SetStats(0)
Canvas.cd(4)
pdfq[0].SetTitle("Student t & its quantiles")
pdf.SetTitle("")
pdf.Draw()
pdfq[0].SetTitle("Student t & its quantiles")
for i in range(9):
    pdfq[i].SetStats(0)
    pdfq[i].SetFillColor(i+1)
    pdfq[i].Draw("same")
Canvas.Modified()
Canvas.cd()
