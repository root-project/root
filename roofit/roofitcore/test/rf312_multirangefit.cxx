//////////////////////////////////////////////////////////////////////////
//
// 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #312
//
// Performing fits in multiple (disjoint) ranges in one or more dimensions
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
#include "RooProdPdf.h"
#include "RooAddPdf.h"
#include "RooPolynomial.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "RooFitResult.h"
using namespace RooFit ;


class TestBasic312 : public RooFitTestUnit
{
public:
  TestBasic312(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Fit in multiple rectangular ranges",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // C r e a t e   2 D   p d f   a n d   d a t a
  // -------------------------------------------

  // Define observables x,y
  RooRealVar x("x","x",-10,10) ;
  RooRealVar y("y","y",-10,10) ;

  // Construct the signal pdf gauss(x)*gauss(y)
  RooRealVar mx("mx","mx",1,-10,10) ;
  RooRealVar my("my","my",1,-10,10) ;

  RooGaussian gx("gx","gx",x,mx,RooConst(1)) ;
  RooGaussian gy("gy","gy",y,my,RooConst(1)) ;

  RooProdPdf sig("sig","sig",gx,gy) ;

  // Construct the background pdf (flat in x,y)
  RooPolynomial px("px","px",x) ;
  RooPolynomial py("py","py",y) ;
  RooProdPdf bkg("bkg","bkg",px,py) ;

  // Construct the composite model sig+bkg
  RooRealVar f("f","f",0.,1.) ;
  RooAddPdf model("model","model",RooArgList(sig,bkg),f) ;

  // Sample 10000 events in (x,y) from the model
  RooDataSet* modelData = model.generate(RooArgSet(x,y),10000) ;



  // D e f i n e   s i g n a l   a n d   s i d e b a n d   r e g i o n s
  // -------------------------------------------------------------------

  // Construct the SideBand1,SideBand2,Signal regions
  //
  //                    |
  //      +-------------+-----------+
  //      |             |           |
  //      |    Side     |   Sig     |
  //      |    Band1    |   nal     |
  //      |             |           |
  //    --+-------------+-----------+--
  //      |                         |
  //      |           Side          |
  //      |           Band2         |
  //      |                         |
  //      +-------------+-----------+
  //                    |

  x.setRange("SB1",-10,+10) ;
  y.setRange("SB1",-10,0) ;

  x.setRange("SB2",-10,0) ;
  y.setRange("SB2",0,+10) ;

  x.setRange("SIG",0,+10) ;
  y.setRange("SIG",0,+10) ;

  x.setRange("FULL",-10,+10) ;
  y.setRange("FULL",-10,+10) ;


  // P e r f o r m   f i t s   i n   i n d i v i d u a l   s i d e b a n d   r e g i o n s
  // -------------------------------------------------------------------------------------

  // Perform fit in SideBand1 region (RooAddPdf coefficients will be interpreted in full range)
  RooFitResult* r_sb1 = model.fitTo(*modelData,Range("SB1"),Save()) ;

  // Perform fit in SideBand2 region (RooAddPdf coefficients will be interpreted in full range)
  RooFitResult* r_sb2 = model.fitTo(*modelData,Range("SB2"),Save()) ;



  // P e r f o r m   f i t s   i n   j o i n t    s i d e b a n d   r e g i o n s
  // -----------------------------------------------------------------------------

  // Now perform fit to joint 'L-shaped' sideband region 'SB1|SB2'
  // (RooAddPdf coefficients will be interpreted in full range)
  RooFitResult* r_sb12 = model.fitTo(*modelData,Range("SB1,SB2"),Save()) ;


  regResult(r_sb1,"rf312_fit_sb1") ;
  regResult(r_sb2,"rf312_fit_sb2") ;
  regResult(r_sb12,"rf312_fit_sb12") ;

  delete modelData ;

  return kTRUE ;

  }
} ;
