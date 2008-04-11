#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooPolynomial.h"
#include "TLine.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic15 : public RooFitTestUnit
{
public: 
  TestBasic15(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {
    
    // "Likelihood ratio projections"

    // Build signal PDF:  gauss(x)*gauss(y)*gauss(z)
    RooRealVar x("x","x",-5,5) ;
    RooRealVar y("y","y",-10,10) ;
    RooRealVar z("z","z",-10,10) ;
    
    RooRealVar meanx("meanx","mean of gaussian x",0) ;
    RooRealVar meany("meany","mean of gaussian y",0) ;
    RooRealVar meanz("meanz","mean of gaussian z",0) ;
    RooRealVar sigmax("sigmax","width of gaussian x",1.25) ;
    RooRealVar sigmay("sigmay","width of gaussian y",2.20) ;
    RooRealVar sigmaz("sigmaz","width of gaussian z",0.70) ;
    RooGaussian gaussx("gaussx","gaussian PDF",x,meanx,sigmax) ;
    RooGaussian gaussy("gaussy","gaussian PDF",y,meany,sigmay) ;
    RooGaussian gaussz("gaussz","gaussian PDF",z,meanz,sigmaz) ;
    
    // Build background PDF: (1+s*x)(1+s*y)(1+s*z)
    RooRealVar slope("slope","slope",0.05) ;
    RooPolynomial polyx("polyx","flat x bkg",x,slope) ;
    RooPolynomial polyy("polyy","flat y bkg",y,slope) ;
    RooPolynomial polyz("polyz","flat z bkg",z,slope) ;
    
    // Build sum pdf: sum=f*sig + (1-f)*bkg
    RooProdPdf sig("sig","sig",RooArgList(gaussx,gaussy,gaussz)) ;
    RooProdPdf bkg("bkg","bkg",RooArgList(polyx,polyy,polyz)) ;
    RooRealVar sigfrac("sigfrac","sigfrac",0.05) ;
    RooAddPdf sum("sum","sig+bkg",RooArgList(sig,bkg),sigfrac) ;
    
    // Create toyMC data set
    RooDataSet* data = sum.generate(RooArgSet(x,y,z),50000) ;
    
    // Calculate likelihood of (y,z) projection of PDF for each event in the dataset
    RooAbsReal* pdfProj=sum.createProjection(x) ;
    RooFormulaVar nllFunc("nll","-log(likelihood)","-log(@0)",*pdfProj) ;
    RooRealVar*   nll = (RooRealVar*) data->addColumn(nllFunc) ;
    
    // Plot x distribution of all events
    RooPlot* xframe1 = x.frame(40) ;
    data->plotOn(xframe1) ;
    sum.plotOn(xframe1) ;
    sum.plotOn(xframe1,Components("bkg"),LineStyle(kDashed),Name("curve_bkg")) ;
    xframe1->SetTitle("All events") ;
    
    // Plot distribution of NLL for all events
    RooPlot* pframe = nll->frame(4,8,100) ;
    data->plotOn(pframe) ;
    pframe->SetTitle("NLL of (y,z) projection of PDF") ;
    
    // Select data based on NLL
    RooDataSet* sliceData = (RooDataSet*) data->reduce(RooArgSet(x,y,z),"nll<5.2") ;
    
    // Plot x distribution for events with NLL<5.2
    RooPlot* xframe2 = x.frame(40) ;
    sliceData->plotOn(xframe2) ;
    sum.plotOn(xframe2,ProjWData(*sliceData)) ;
    sum.plotOn(xframe2,Components("bkg"),ProjWData(*sliceData),LineStyle(kDashed),Name("curve_bkg")) ;
    xframe2->SetTitle("Events with NLL<5.2") ;

    regPlot(xframe1,"Basic15_PlotX1") ;
    regPlot(pframe ,"Basic15_PlotP") ;
    regPlot(xframe2,"Basic15_PlotX2") ;
    
    delete sliceData ;
    delete data ;
    delete pdfProj ;

    return kTRUE ;
  }
} ;
