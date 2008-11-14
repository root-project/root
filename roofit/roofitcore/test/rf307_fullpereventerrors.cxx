//////////////////////////////////////////////////////////////////////////
//
// 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #307
// 
// Complete example with use of full p.d.f. with per-event errors
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
#include "RooProdPdf.h"
#include "RooHistPdf.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


class TestBasic307 : public RooFitTestUnit
{
public: 
  TestBasic307(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Full per-event error p.d.f. F(t|dt)G(dt)",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // B - p h y s i c s   p d f   w i t h   p e r - e v e n t  G a u s s i a n   r e s o l u t i o n
  // ----------------------------------------------------------------------------------------------

  // Observables
  RooRealVar dt("dt","dt",-10,10) ;
  RooRealVar dterr("dterr","per-event error on dt",0.1,10) ;

  // Build a gaussian resolution model scaled by the per-event error = gauss(dt,bias,sigma*dterr)
  RooRealVar bias("bias","bias",0,-10,10) ;
  RooRealVar sigma("sigma","per-event error scale factor",1,0.1,10) ;
  RooGaussModel gm("gm1","gauss model scaled bt per-event error",dt,bias,sigma,dterr) ;

  // Construct decay(dt) (x) gauss1(dt|dterr)
  RooRealVar tau("tau","tau",1.548) ;
  RooDecay decay_gm("decay_gm","decay",dt,tau,gm,RooDecay::DoubleSided) ;



  // C o n s t r u c t   e m p i r i c a l   p d f   f o r   p e r - e v e n t   e r r o r
  // -----------------------------------------------------------------

  // Use landau p.d.f to get empirical distribution with long tail
  RooLandau pdfDtErr("pdfDtErr","pdfDtErr",dterr,RooConst(1),RooConst(0.25)) ;
  RooDataSet* expDataDterr = pdfDtErr.generate(dterr,10000) ;

  // Construct a histogram pdf to describe the shape of the dtErr distribution
  RooDataHist* expHistDterr = expDataDterr->binnedClone() ;
  RooHistPdf pdfErr("pdfErr","pdfErr",dterr,*expHistDterr) ;


  // C o n s t r u c t   c o n d i t i o n a l   p r o d u c t   d e c a y _ d m ( d t | d t e r r ) * p d f ( d t e r r )
  // ----------------------------------------------------------------------------------------------------------------------

  // Construct production of conditional decay_dm(dt|dterr) with empirical pdfErr(dterr)
  RooProdPdf model("model","model",pdfErr,Conditional(decay_gm,dt)) ;

  // (Alternatively you could also use the landau shape pdfDtErr)
  //RooProdPdf model("model","model",pdfDtErr,Conditional(decay_gm,dt)) ;

  

  // S a m p l e,   f i t   a n d   p l o t   p r o d u c t   m o d e l 
  // ------------------------------------------------------------------

  // Specify external dataset with dterr values to use model_dm as conditional p.d.f.
  RooDataSet* data = model.generate(RooArgSet(dt,dterr),10000) ;

  

  // F i t   c o n d i t i o n a l   d e c a y _ d m ( d t | d t e r r )
  // ---------------------------------------------------------------------

  // Specify dterr as conditional observable
  model.fitTo(*data) ;


  
  // P l o t   c o n d i t i o n a l   d e c a y _ d m ( d t | d t e r r )
  // ---------------------------------------------------------------------


  // Make projection of data an dt
  RooPlot* frame = dt.frame(Title("Projection of model(dt|dterr) on dt")) ;
  data->plotOn(frame) ;
  model.plotOn(frame) ;


  regPlot(frame,"rf307_plot1") ;

  delete expDataDterr ;
  delete expHistDterr ;
  delete data ;
  
  return kTRUE ;

  }
} ;
