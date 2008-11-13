//////////////////////////////////////////////////////////////////////////
//
// 'DATA AND CATEGORIES' RooFit tutorial macro #402
// 
// Tools for manipulation of (un)binned datasets
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
#include "RooDataHist.h"
#include "RooGaussian.h"
#include "RooCategory.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "TFile.h"
using namespace RooFit ;


class TestBasic402 : public RooFitTestUnit
{
public: 
  TestBasic402(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Basic operations on datasets",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // Binned (RooDataHist) and unbinned datasets (RooDataSet) share
  // many properties and inherit from a common abstract base class
  // (RooAbsData), that provides an interface for all operations
  // that can be performed regardless of the data format

  RooRealVar  x("x","x",-10,10) ;
  RooRealVar  y("y","y", 0, 40) ;
  RooCategory c("c","c") ;
  c.defineType("Plus",+1) ;
  c.defineType("Minus",-1) ;



  // B a s i c   O p e r a t i o n s   o n   u n b i n n e d   d a t a s e t s 
  // --------------------------------------------------------------

  // RooDataSet is an unbinned dataset (a collection of points in N-dimensional space)
  RooDataSet d("d","d",RooArgSet(x,y,c)) ;

  // Unlike RooAbsArgs (RooAbsPdf,RooFormulaVar,....) datasets are not attached to 
  // the variables they are constructed from. Instead they are attached to an internal 
  // clone of the supplied set of arguments

  // Fill d with dummy values
  Int_t i ;
  for (i=0 ; i<1000 ; i++) {
    x = i/50 - 10 ;
    y = sqrt(1.0*i) ;
    c.setLabel((i%2)?"Plus":"Minus") ;

    // We must explicitly refer to x,y,c here to pass the values because
    // d is not linked to them (as explained above)
    d.add(RooArgSet(x,y,c)) ;
  }


  // R e d u c i n g ,   A p p e n d i n g   a n d   M e r g i n g
  // -------------------------------------------------------------

  // The reduce() function returns a new dataset which is a subset of the original
  RooDataSet* d1 = (RooDataSet*) d.reduce(RooArgSet(x,c)) ; 
  RooDataSet* d2 = (RooDataSet*) d.reduce(RooArgSet(y)) ;   
  RooDataSet* d3 = (RooDataSet*) d.reduce("y>5.17") ; 
  RooDataSet* d4 = (RooDataSet*) d.reduce(RooArgSet(x,c),"y>5.17") ; 

  regValue(d3->numEntries(),"rf403_nd3") ;
  regValue(d4->numEntries(),"rf403_nd4") ;

  // The merge() function adds two data set column-wise
  d1->merge(d2) ;

  // The append() function addes two datasets row-wise
  d1->append(*d3) ;

  regValue(d1->numEntries(),"rf403_nd1") ;

  


  // O p e r a t i o n s   o n   b i n n e d   d a t a s e t s 
  // ---------------------------------------------------------

  // A binned dataset can be constructed empty, from an unbinned dataset, or
  // from a ROOT native histogram (TH1,2,3)

  // The binning of real variables (like x,y) is done using their fit range
  // 'get/setRange()' and number of specified fit bins 'get/setBins()'.
  // Category dimensions of binned datasets get one bin per defined category state
  x.setBins(10) ;
  y.setBins(10) ;
  RooDataHist dh("dh","binned version of d",RooArgSet(x,y),d) ;

  RooPlot* yframe = y.frame(Bins(10),Title("Operations on binned datasets")) ;
  dh.plotOn(yframe) ; // plot projection of 2D binned data on y

  // Reduce the 2-dimensional binned dataset to a 1-dimensional binned dataset
  //
  // All reduce() methods are interfaced in RooAbsData. All reduction techniques
  // demonstrated on unbinned datasets can be applied to binned datasets as well.
  RooDataHist* dh2 = (RooDataHist*) dh.reduce(y,"x>0") ;

  // Add dh2 to yframe and redraw
  dh2->plotOn(yframe,LineColor(kRed),MarkerColor(kRed),Name("dh2")) ;

  regPlot(yframe,"rf402_plot1") ;

  delete d1 ;
  delete d2 ;
  delete d3 ;
  delete d4 ;
  delete dh2 ;
  return kTRUE ;
  }
} ;
