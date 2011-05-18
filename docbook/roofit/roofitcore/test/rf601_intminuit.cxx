/////////////////////////////////////////////////////////////////////////
//
// 'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #601
// 
// Interactive minimization with MINUIT
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
#include "RooProdPdf.h"
#include "RooAddPdf.h"
#include "RooMinuit.h"
#include "RooNLLVar.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


class TestBasic601 : public RooFitTestUnit
{
public: 
  TestBasic601(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Interactive Minuit",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // S e t u p   p d f   a n d   l i k e l i h o o d 
  // -----------------------------------------------

  // Observable
  RooRealVar x("x","x",-20,20) ;

  // Model (intentional strong correlations)
  RooRealVar mean("mean","mean of g1 and g2",0) ;
  RooRealVar sigma_g1("sigma_g1","width of g1",3) ;
  RooGaussian g1("g1","g1",x,mean,sigma_g1) ;

  RooRealVar sigma_g2("sigma_g2","width of g2",4,3.0,6.0) ;
  RooGaussian g2("g2","g2",x,mean,sigma_g2) ;

  RooRealVar frac("frac","frac",0.5,0.0,1.0) ;
  RooAddPdf model("model","model",RooArgList(g1,g2),frac) ;

  // Generate 1000 events
  RooDataSet* data = model.generate(x,1000) ;
  
  // Construct unbinned likelihood
  RooNLLVar nll("nll","nll",model,*data) ;


  // I n t e r a c t i v e   m i n i m i z a t i o n ,   e r r o r   a n a l y s i s
  // -------------------------------------------------------------------------------

  // Create MINUIT interface object
  RooMinuit m(nll) ;

  // Call MIGRAD to minimize the likelihood
  m.migrad() ;

  // Run HESSE to calculate errors from d2L/dp2
  m.hesse() ;

  // Run MINOS on sigma_g2 parameter only
  m.minos(sigma_g2) ;


  // S a v i n g   r e s u l t s ,   c o n t o u r   p l o t s 
  // ---------------------------------------------------------

  // Save a snapshot of the fit result. This object contains the initial
  // fit parameters, the final fit parameters, the complete correlation
  // matrix, the EDM, the minimized FCN , the last MINUIT status code and
  // the number of times the RooFit function object has indicated evaluation
  // problems (e.g. zero probabilities during likelihood evaluation)
  RooFitResult* r = m.save() ;


  // C h a n g e   p a r a m e t e r   v a l u e s ,   f l o a t i n g
  // -----------------------------------------------------------------

  // At any moment you can manually change the value of a (constant)
  // parameter
  mean = 0.3 ;
  
  // Rerun MIGRAD,HESSE
  m.migrad() ;
  m.hesse() ;

  // Now fix sigma_g2
  sigma_g2.setConstant(kTRUE) ;

  // Rerun MIGRAD,HESSE
  m.migrad() ;
  m.hesse() ;

  RooFitResult* r2 = m.save() ;

  regResult(r,"rf601_r") ;
  regResult(r2,"rf601_r2") ;

  delete data ;

  return kTRUE ;
  } 
} ;
