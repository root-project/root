//////////////////////////////////////////////////////////////////////////
//
// 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #306
//
// Complete example with use of conditional p.d.f. with per-event errors
//
//
//
// 07/2008 - Wouter Verkerke
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooGaussModel.h"
#include "RooDecay.h"
#include "RooLandau.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH2D.h"
using namespace RooFit ;



class TestBasic306 : public RooFitTestUnit
{
public:
  TestBasic306(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("Conditional use of per-event error p.d.f. F(t|dt)",refFile,writeRef,verbose) {} ;
  bool testCode() {

  // B - p h y s i c s   p d f   w i t h   p e r - e v e n t  G a u s s i a n   r e s o l u t i o n
  // ----------------------------------------------------------------------------------------------

  // Observables
  RooRealVar dt("dt","dt",-10,10) ;
  RooRealVar dterr("dterr","per-event error on dt",0.01,10) ;

  // Build a gaussian resolution model scaled by the per-event error = gauss(dt,bias,sigma*dterr)
  RooRealVar bias("bias","bias",0,-10,10) ;
  RooRealVar sigma("sigma","per-event error scale factor",1,0.1,10) ;
  RooGaussModel gm("gm1","gauss model scaled bt per-event error",dt,bias,sigma,dterr) ;

  // Construct decay(dt) (x) gauss1(dt|dterr)
  RooRealVar tau("tau","tau",1.548) ;
  RooDecay decay_gm("decay_gm","decay",dt,tau,gm,RooDecay::DoubleSided) ;



  // C o n s t r u c t   f a k e   ' e x t e r n a l '   d a t a    w i t h   p e r - e v e n t   e r r o r
  // ------------------------------------------------------------------------------------------------------

  // Use landau p.d.f to get somewhat realistic distribution with long tail
  RooLandau pdfDtErr("pdfDtErr","pdfDtErr",dterr,RooConst(1),RooConst(0.25)) ;
  RooDataSet* expDataDterr = pdfDtErr.generate(dterr,10000) ;



  // S a m p l e   d a t a   f r o m   c o n d i t i o n a l   d e c a y _ g m ( d t | d t e r r )
  // ---------------------------------------------------------------------------------------------

  // Specify external dataset with dterr values to use decay_dm as conditional p.d.f.
  RooDataSet* data = decay_gm.generate(dt,ProtoData(*expDataDterr)) ;



  // F i t   c o n d i t i o n a l   d e c a y _ d m ( d t | d t e r r )
  // ---------------------------------------------------------------------

  // Specify dterr as conditional observable
  decay_gm.fitTo(*data,ConditionalObservables(dterr)) ;



  // P l o t   c o n d i t i o n a l   d e c a y _ d m ( d t | d t e r r )
  // ---------------------------------------------------------------------


  // Make two-dimensional plot of conditional p.d.f in (dt,dterr)
  TH1* hh_decay = decay_gm.createHistogram("hh_decay",dt,Binning(50),YVar(dterr,Binning(50))) ;
  hh_decay->SetLineColor(kBlue) ;


  // Plot decay_gm(dt|dterr) at various values of dterr
  RooPlot* frame = dt.frame(Title("Slices of decay(dt|dterr) at various dterr")) ;
  for (Int_t ibin=0 ; ibin<100 ; ibin+=20) {
    dterr.setBin(ibin) ;
    decay_gm.plotOn(frame,Normalization(5.),Name(Form("curve_slice_%d",ibin))) ;
  }


  // Make projection of data an dt
  RooPlot* frame2 = dt.frame(Title("Projection of decay(dt|dterr) on dt")) ;
  data->plotOn(frame2) ;

  // Make projection of decay(dt|dterr) on dt.
  //
  // Instead of integrating out dterr, make a weighted average of curves
  // at values dterr_i as given in the external dataset.
  // (The true argument bins the data before projection to speed up the process)
  decay_gm.plotOn(frame2,ProjWData(*expDataDterr,true)) ;


  regTH(hh_decay,"rf306_model2d") ;
  regPlot(frame,"rf306_plot1") ;
  regPlot(frame2,"rf306_plot2") ;

  delete expDataDterr ;
  delete data ;

  return true ;
  }
} ;

