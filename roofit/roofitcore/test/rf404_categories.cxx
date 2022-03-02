//////////////////////////////////////////////////////////////////////////
//
// 'DATA AND CATEGORIES' RooFit tutorial macro #404
//
// Working with RooCategory objects to describe discrete variables
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
#include "RooPolynomial.h"
#include "RooCategory.h"
#include "Roo1DTable.h"
#include "RooGaussian.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


class TestBasic404 : public RooFitTestUnit
{
public:
  TestBasic404(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Categories basic functionality",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // C o n s t r u c t    a   c a t e g o r y   w i t h   l a b e l s
  // ----------------------------------------------------------------

  // Define a category with labels only
  RooCategory tagCat("tagCat","Tagging category") ;
  tagCat.defineType("Lepton") ;
  tagCat.defineType("Kaon") ;
  tagCat.defineType("NetTagger-1") ;
  tagCat.defineType("NetTagger-2") ;



  // C o n s t r u c t    a   c a t e g o r y   w i t h   l a b e l s   a n d   i n d e c e s
  // ----------------------------------------------------------------------------------------

  // Define a category with explicitly numbered states
  RooCategory b0flav("b0flav","B0 flavour eigenstate") ;
  b0flav.defineType("B0",-1) ;
  b0flav.defineType("B0bar",1) ;



  // G e n e r a t e   d u m m y   d a t a  f o r   t a b u l a t i o n   d e m o
  // ----------------------------------------------------------------------------

  // Generate a dummy dataset
  RooRealVar x("x","x",0,10) ;
  RooDataSet *data = RooPolynomial("p","p",x).generate(RooArgSet(x,b0flav,tagCat),10000) ;



  // P r i n t   t a b l e s   o f   c a t e g o r y   c o n t e n t s   o f   d a t a s e t s
  // ------------------------------------------------------------------------------------------

  // Tables are equivalent of plots for categories
  Roo1DTable* btable = data->table(b0flav) ;

  // Create table for subset of events matching cut expression
  Roo1DTable* ttable = data->table(tagCat,"x>8.23") ;

  // Create table for all (tagCat x b0flav) state combinations
  Roo1DTable* bttable = data->table(RooArgSet(tagCat,b0flav)) ;

  // Retrieve number of events from table
  // Number can be non-integer if source dataset has weighed events
  Double_t nb0 = btable->get("B0") ;
  regValue(nb0,"rf404_nb0") ;

  // Retrieve fraction of events with "Lepton" tag
  Double_t fracLep = ttable->getFrac("Lepton") ;
  regValue(fracLep,"rf404_fracLep") ;


  // D e f i n i n g   r a n g e s   f o r   p l o t t i n g ,   f i t t i n g   o n   c a t e g o r i e s
  // ------------------------------------------------------------------------------------------------------

  // Define named range as comma separated list of labels
  tagCat.setRange("good","Lepton,Kaon") ;

  // Or add state names one by one
  tagCat.addToRange("soso","NetTagger-1") ;
  tagCat.addToRange("soso","NetTagger-2") ;

  // Use category range in dataset reduction specification
  RooDataSet* goodData = (RooDataSet*) data->reduce(CutRange("good")) ;
  Roo1DTable* gtable = goodData->table(tagCat) ;


  regTable(btable,"rf404_btable") ;
  regTable(ttable,"rf404_ttable") ;
  regTable(bttable,"rf404_bttable") ;
  regTable(gtable,"rf404_gtable") ;


  delete goodData ;
  delete data ;
  return kTRUE ;

  }

} ;
