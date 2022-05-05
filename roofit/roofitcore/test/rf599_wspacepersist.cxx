//////////////////////////////////////////////////////////////////////////
//
// 'ORGANIZATION AND SIMULTANEOUS FITS' RooFit tutorial macro #501
//
// Using simultaneous p.d.f.s to describe simultaneous fits to multiple
// datasets
//
//
//
// 07/2008 - Wouter Verkerke
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooWorkspace.h"
#include "RooProdPdf.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooGaussModel.h"
#include "RooAddModel.h"
#include "RooDecay.h"
#include "RooChebychev.h"
#include "RooAddPdf.h"
#include "RooSimultaneous.h"
#include "RooCategory.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


class TestBasic599 : public RooFitTestUnit
{
public:
  TestBasic599(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("Workspace and p.d.f. persistence",refFile,writeRef,verbose) {} ;
  bool testCode() {

    if (_write) {

      RooWorkspace *w = new RooWorkspace("TestBasic11_ws") ;

      regWS(w,"Basic11_ws") ;

      // Build Gaussian PDF in X
      RooRealVar x("x","x",-10,10) ;
      RooRealVar meanx("meanx","mean of gaussian",-1) ;
      RooRealVar sigmax("sigmax","width of gaussian",3) ;
      RooGaussian gaussx("gaussx","gaussian PDF",x,meanx,sigmax) ;

      // Build Gaussian PDF in Y
      RooRealVar y("y","y",-10,10) ;
      RooRealVar meany("meany","mean of gaussian",-1) ;
      RooRealVar sigmay("sigmay","width of gaussian",3) ;
      RooGaussian gaussy("gaussy","gaussian PDF",y,meany,sigmay) ;

      // Make product of X and Y
      RooProdPdf gaussxy("gaussxy","gaussx*gaussy",RooArgSet(gaussx,gaussy)) ;

      // Make flat bkg in X and Y
      RooPolynomial flatx("flatx","flatx",x) ;
      RooPolynomial flaty("flaty","flaty",x) ;
      RooProdPdf flatxy("flatxy","flatx*flaty",RooArgSet(flatx,flaty)) ;

      // Make sum of gaussxy and flatxy
      RooRealVar frac("frac","frac",0.5,0.,1.) ;
      RooAddPdf sumxy("sumxy","sumxy",RooArgList(gaussxy,flatxy),frac) ;

      // Store p.d.f in workspace
      w->import(gaussx) ;
      w->import(gaussxy,RenameConflictNodes("set2")) ;
      w->import(sumxy,RenameConflictNodes("set3")) ;

      // Make reference plot of GaussX
      RooPlot* frame1 = x.frame() ;
      gaussx.plotOn(frame1) ;
      regPlot(frame1,"Basic11_gaussx_framex") ;

      // Make reference plots for GaussXY
      RooPlot* frame2 = x.frame() ;
      gaussxy.plotOn(frame2) ;
      regPlot(frame2,"Basic11_gaussxy_framex") ;

      RooPlot* frame3 = y.frame() ;
      gaussxy.plotOn(frame3) ;
      regPlot(frame3,"Basic11_gaussxy_framey") ;

      // Make reference plots for SumXY
      RooPlot* frame4 = x.frame() ;
      sumxy.plotOn(frame4) ;
      regPlot(frame4,"Basic11_sumxy_framex") ;

      RooPlot* frame5 = y.frame() ;
      sumxy.plotOn(frame5) ;
      regPlot(frame5,"Basic11_sumxy_framey") ;

      // Analytically convolved p.d.f.s

      // Build a simple decay PDF
      RooRealVar dt("dt","dt",-20,20) ;
      RooRealVar tau("tau","tau",1.548) ;

      // Build a gaussian resolution model
      RooRealVar bias1("bias1","bias1",0) ;
      RooRealVar sigma1("sigma1","sigma1",1) ;
      RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1) ;

      // Construct a decay PDF, smeared with single gaussian resolution model
      RooDecay decay_gm1("decay_gm1","decay",dt,tau,gm1,RooDecay::DoubleSided) ;

      // Build another gaussian resolution model
      RooRealVar bias2("bias2","bias2",0) ;
      RooRealVar sigma2("sigma2","sigma2",5) ;
      RooGaussModel gm2("gm2","gauss model 2",dt,bias2,sigma2) ;

      // Build a composite resolution model
      RooRealVar gm1frac("gm1frac","fraction of gm1",0.5) ;
      RooAddModel gmsum("gmsum","sum of gm1 and gm2",RooArgList(gm1,gm2),gm1frac) ;

      // Construct a decay PDF, smeared with double gaussian resolution model
      RooDecay decay_gmsum("decay_gmsum","decay",dt,tau,gmsum,RooDecay::DoubleSided) ;

      w->import(decay_gm1) ;
      w->import(decay_gmsum,RenameConflictNodes("set3")) ;

      RooPlot* frame6 = dt.frame() ;
      decay_gm1.plotOn(frame6) ;
      regPlot(frame6,"Basic11_decay_gm1_framedt") ;

      RooPlot* frame7 = dt.frame() ;
      decay_gmsum.plotOn(frame7) ;
      regPlot(frame7,"Basic11_decay_gmsum_framedt") ;

      // Construct simultaneous p.d.f
      RooCategory cat("cat","cat") ;
      cat.defineType("A") ;
      cat.defineType("B") ;
      RooSimultaneous sim("sim","sim",cat) ;
      sim.addPdf(gaussxy,"A") ;
      sim.addPdf(flatxy,"B") ;

      w->import(sim,RenameConflictNodes("set4")) ;

      // Make plot with dummy dataset for index projection
      RooDataHist dh("dh","dh",cat) ;
      cat.setLabel("A") ;
      dh.add(cat) ;
      cat.setLabel("B") ;
      dh.add(cat) ;

      RooPlot* frame8 = x.frame() ;
      sim.plotOn(frame8,ProjWData(cat,dh),Project(cat)) ;

      regPlot(frame8,"Basic11_sim_framex") ;


    } else {

      RooWorkspace* w = getWS("Basic11_ws") ;
      if (!w) return false ;

      // Retrieve p.d.f from workspace
      RooAbsPdf* gaussx = w->pdf("gaussx") ;

      // Make test plot and offer for comparison against ref plot
      RooPlot* frame1 = w->var("x")->frame() ;
      gaussx->plotOn(frame1) ;
      regPlot(frame1,"Basic11_gaussx_framex") ;

      // Retrieve p.d.f from workspace
      RooAbsPdf* gaussxy = w->pdf("gaussxy") ;

      // Make test plot and offer for comparison against ref plot
      RooPlot* frame2 = w->var("x")->frame() ;
      gaussxy->plotOn(frame2) ;
      regPlot(frame2,"Basic11_gaussxy_framex") ;

      // Make test plot and offer for comparison against ref plot
      RooPlot* frame3 = w->var("y")->frame() ;
      gaussxy->plotOn(frame3) ;
      regPlot(frame3,"Basic11_gaussxy_framey") ;

      // Retrieve p.d.f from workspace
      RooAbsPdf* sumxy = w->pdf("sumxy") ;

      // Make test plot and offer for comparison against ref plot
      RooPlot* frame4 = w->var("x")->frame() ;
      sumxy->plotOn(frame4) ;
      regPlot(frame4,"Basic11_sumxy_framex") ;

      // Make test plot and offer for comparison against ref plot
      RooPlot* frame5 = w->var("y")->frame() ;
      sumxy->plotOn(frame5) ;
      regPlot(frame5,"Basic11_sumxy_framey") ;

      // Retrieve p.d.f from workspace
      RooAbsPdf* decay_gm1 = w->pdf("decay_gm1") ;

      // Make test plot and offer for comparison against ref plot
      RooPlot* frame6 = w->var("dt")->frame() ;
      decay_gm1->plotOn(frame6) ;
      regPlot(frame6,"Basic11_decay_gm1_framedt") ;

      // Retrieve p.d.f from workspace
      RooAbsPdf* decay_gmsum = w->pdf("decay_gmsum") ;

      // Make test plot and offer for comparison against ref plot
      RooPlot* frame7 = w->var("dt")->frame() ;
      decay_gmsum->plotOn(frame7) ;
      regPlot(frame7,"Basic11_decay_gmsum_framedt") ;

      // Retrieve p.d.f. from workspace
      RooAbsPdf* sim = w->pdf("sim") ;
      RooCategory* cat = w->cat("cat") ;

      // Make plot with dummy dataset for index projection
      RooPlot* frame8 = w->var("x")->frame() ;

      RooDataHist dh("dh","dh",*cat) ;
      cat->setLabel("A") ;
      dh.add(*cat) ;
      cat->setLabel("B") ;
      dh.add(*cat) ;

      sim->plotOn(frame8,ProjWData(*cat,dh),Project(*cat)) ;

      regPlot(frame8,"Basic11_sim_framex") ;

    }

    // "Workspace persistence"
    return true ;
  }
} ;
