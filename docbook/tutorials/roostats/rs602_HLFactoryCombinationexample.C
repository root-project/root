/////////////////////////////////////////////////////////////////////////
//
// 'High Level Factory Example' RooStats tutorial macro #602
// author: Danilo Piparo
// date August. 2009
//
// This tutorial shows an example of creating a combined
// model using the High Level model Factory.
//
//
/////////////////////////////////////////////////////////////////////////

#include <fstream>
#include "TString.h"
#include "TROOT.h"
#include "RooGlobalFunc.h"
#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooStats/HLFactory.h"



// use this order for safety on library loading
using namespace RooFit ;
using namespace RooStats ;
using namespace std;

void rs602_HLFactoryCombinationexample() {

using namespace RooStats;
using namespace RooFit;

// create a card
TString card_name("HLFavtoryCombinationexample.rs");
ofstream ofile(card_name);
ofile << "// The simplest card for combination\n\n"
      << "gauss1 = Gaussian(x[0,100],mean1[50,0,100],4);\n"
      << "flat1 = Polynomial(x,0);\n"
      << "sb_model1 = SUM(nsig1[120,0,300]*gauss1 , nbkg1[100,0,1000]*flat1);\n"
      << "gauss2 = Gaussian(x,mean2[80,0,100],5);\n"
      << "flat2 = Polynomial(x,0);\n"
      << "sb_model2 = SUM(nsig2[90,0,400]*gauss2 , nbkg2[80,0,1000]*flat2);\n";

ofile.close();

HLFactory hlf("HLFavtoryCombinationexample",
              card_name,
              false);

hlf.AddChannel("model1","sb_model1","flat1");
hlf.AddChannel("model2","sb_model2","flat2");
RooAbsPdf* pdf=hlf.GetTotSigBkgPdf();
RooCategory* thecat = hlf.GetTotCategory();
RooRealVar* x= static_cast<RooRealVar*>(hlf.GetWs()->arg("x"));

RooDataSet* data = pdf->generate(RooArgSet(*x,*thecat),Extended());

// --- Perform extended ML fit of composite PDF to toy data ---
pdf->fitTo(*data) ;

// --- Plot toy data and composite PDF overlaid ---
RooPlot* xframe = x->frame() ;

data->plotOn(xframe);
thecat->setIndex(0);
pdf->plotOn(xframe,Slice(*thecat),ProjWData(*thecat,*data)) ;

thecat->setIndex(1);
pdf->plotOn(xframe,Slice(*thecat),ProjWData(*thecat,*data)) ;

gROOT->SetStyle("Plain");
xframe->Draw();
}
