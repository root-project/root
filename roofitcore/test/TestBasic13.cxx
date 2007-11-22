#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooGlobalFunc.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic13 : public RooFitTestUnit
{
public: 
  TestBasic13(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Data import and persistence"

    {
      // Binned (RooDataHist) and unbinned datasets (RooDataSet) share
      // many properties and inherit from a common abstract base class
      // (RooAbsData), that provides an interface for all operations
      // that can be performed regardless of the data format
      
      RooRealVar  x("x","x",-10,10) ;
      RooRealVar  y("y","y", 0, 40) ;
      RooCategory c("c","c") ;
      c.defineType("Plus",+1) ;
      c.defineType("Minus",-1) ;
      
      // *** Unbinned datasets ***
      
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
      
      // *** Reducing / Appending / Merging ***
      
      // The reduce() function returns a new dataset which is a subset of the original
      RooDataSet* d1 = (RooDataSet*) d.reduce(RooArgSet(x,c)) ; 
      
      RooDataSet* d2 = (RooDataSet*) d.reduce(RooArgSet(y)) ;   
      
      RooDataSet* d3 = (RooDataSet*) d.reduce("y>5.17") ; 
      
      RooDataSet* d4 = (RooDataSet*) d.reduce(RooArgSet(x,c),"y>5.17") ; 

      regValue(d1->numEntries(),"Basic13_NumD1") ;
      regValue(d2->numEntries(),"Basic13_NumD2") ;
      regValue(d3->numEntries(),"Basic13_NumD3") ;
      regValue(d4->numEntries(),"Basic13_NumD4") ;
      
      // The merge() function adds two data set column-wise
      d1->merge(d2) ; 
      regValue(d4->numEntries(),"Basic13_NumD12") ;
      
      
      // The append() function addes two datasets row-wise
      d1->append(*d3) ;
      regValue(d4->numEntries(),"Basic13_NumD34") ;
      
      // *** Binned datasets ***
      
      // A binned dataset can be constructed empty, from an unbinned dataset, or
      // from a ROOT native histogram (TH1,2,3)
      
      // The binning of real variables (like x,y) is done using their fit range
      //'get/setRange()' and number of specified fit bins 'get/setBins()'.
      // Category dimensions of binned datasets get one bin per defined category state
      x.setBins(10) ;
      y.setBins(10) ;
      RooDataHist dh("dh","binned version of d",RooArgSet(x,y),d) ;
      
      RooPlot* yframe = y.frame(10) ;
      dh.plotOn(yframe) ; // plot projection of 2D binned data on y
      
      // Examine the statistics of a binned dataset
      
      // Locate a bin from a set of coordinates and retrieve its properties
      x = 0.3 ;  y = 20.5 ;
      dh.get(RooArgSet(x,y)) ; // load bin center coordinates in internal buffer
      regValue(dh.weight(),"Basic13_WeightXY") ;
      
      // Reduce the 2-dimensional binned dataset to a 1-dimensional binned dataset
      //
      // All reduce() methods are interfaced in RooAbsData. All reduction techniques
      // demonstrated on unbinned datasets can be applied to binned datasets as well.
      RooDataHist* dh2 = (RooDataHist*) dh.reduce(y,"x>0") ;
      regValue(dh2->numEntries(),"Basic13_Num2Red") ;
      
      // Add dh2 to yframe and redraw
      dh2->plotOn(yframe,LineColor(kRed),MarkerColor(kRed),Name("dh2")) ;

      regPlot(yframe,"Basic13_PlotY") ;

      delete d1 ;
      delete d2 ;
      delete d3 ;
      delete d4 ;
      delete dh2 ;
      
}


    return kTRUE ;
  }
} ;
