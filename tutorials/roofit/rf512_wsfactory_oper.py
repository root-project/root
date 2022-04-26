## \file
## \ingroup tutorial_roofit
## \notebook
##
## 'ORGANIZATION AND SIMULTANEOUS FITS' RooFit tutorial macro #512
##
## Illustration of operator expressions and expression-based
## basic p.d.f.s in the workspace factory syntax
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C version)


import ROOT


w = ROOT.RooWorkspace("w")

# You can define typedefs for even shorter construction semantics
w.factory("$Typedef(Gaussian,Gaus)")
w.factory("$Typedef(Chebychev,Cheby)")

# Operator pdf examples
# ------------------------------------------------

# PDF addition is done with SUM (coef1*pdf1,pdf2)
w.factory("SUM::summodel( f[0,1]*Gaussian::gx(x[-10,10],m[0],1.0), Chebychev::ch(x,{0.1,0.2,-0.3}) )")

# Extended PDF addition is done with SUM (yield1*pdf1,yield2*pdf2)
w.factory("SUM::extsummodel( Nsig[0,1000]*gx, Nbkg[0,1000]*ch )")

# PDF multiplication is done with PROD ( pdf1, pdf2 )
w.factory("PROD::gxz( gx, Gaussian::gz(z[-10,10],0,1) )")

# Conditional p.d.f multiplication is done with PROD ( pdf1|obs, pdf2 )
w.factory("Gaussian::gy( y[-10,10], x, 1.0 )")
w.factory("PROD::gxycond( gy|x, gx )")

# Convolution (numeric/ fft) is done with NCONV/FCONV (obs,pdf1,pdf2)
w.factory("FCONV::lxg( x, Gaussian::g(x,mg[0],1), Landau::lc(x,0,1) )")

# Simultaneous p.d.f.s are constructed with SIMUL( index, state1=pdf1,
# state2=pdf2,...)
w.factory("SIMUL::smodel( c[A=0,B=1], A=Gaussian::gs(x,m,s[1]), B=Landau::ls(x,0,1) )")

# Operator function examples
# ---------------------------------------------------

# Function multiplication is done with prod (func1, func2,...)
w.factory("prod::uv(u[10],v[10])")

# Function addition is done with sum(func1,func2)
w.factory("sum::uv2(u,v)")

# Lagrangian morphing function for the example shown in rf711_lagrangianmorph
infilename = ROOT.gROOT.GetTutorialDir().Data() + "/roofit/input_histos_rf_lagrangianmorph.root"
w.factory(
    "lagrangianmorph::morph($observableName('pTV'),$fileName('"
    + infilename
    + "'),$couplings({cHq3[0,1],SM[1]}),$NewPhysics(cHq3=1,SM=0),$folders({'SM_NPsq0','cHq3_NPsq1','cHq3_NPsq2'}))"
)

# Taylor expansion is done with taylorexpand(func,{var1,var2,...},val,order)
w.factory("taylorexpand::te(expr::poly('x^4+5*x^3+2*x^2+x+1',x),{x},0.0,2)")


# Interpreted and compiled expression based pdfs
# ---------------------------------------------------------------------------------------------------

# Create a ROOT.RooGenericPdf interpreted p.d.f. You can use single quotes
# to pass the expression string argument
w.factory("EXPR::G('x*x+1',x)")

# Create a custom compiled p.d.f similar to the above interpreted p.d.f.
# The code required to make self p.d.f. is automatically embedded in
# the workspace
w.factory("CEXPR::GC('x*x+a',{x,a[1]})")

# Compiled and interpreted functions (rather than p.d.f.s) can be made with the lower case
# 'expr' and 'cexpr' types

# Print workspace contents
w.Print()

# Make workspace visible on command line
ROOT.gDirectory.Add(w)
