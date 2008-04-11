#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooCategory.h"
#include "RooMappedCategory.h"
#include "RooSuperCategory.h"
#include "RooThresholdCategory.h"
#include "Roo1DTable.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic10 : public RooFitTestUnit
{
public: 
  TestBasic10(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Functions of discrete variables"

    // Define a category with explicitly numbered states
    RooCategory b0flav("b0flav","B0 flavour eigenstate") ;
    b0flav.defineType("B0",-1) ;
    b0flav.defineType("B0bar",1) ;
    // b0flav.Print("s") ;
    
    // Define a category with labels only
    RooCategory tagCat("tagCat","Tagging category") ;
    tagCat.defineType("Lepton") ;
    tagCat.defineType("Kaon") ;
    tagCat.defineType("NetTagger-1") ;
    tagCat.defineType("NetTagger-2") ;
    // tagCat.Print("s") ;
    
    // Define a dummy PDF in x
    RooRealVar x("x","x",0,10) ;
    RooArgusBG a("a","argus(x)",x,RooRealConstant::value(10),RooRealConstant::value(-1)) ;
    
    // Generate a dummy dataset
    RooDataSet *data = a.generate(RooArgSet(x,b0flav,tagCat),10000) ;
    
    // Tables are equivalent of plots for categories
    RooTable* btable = data->table(b0flav) ;
    regTable(btable,"Basic10_BTable");
    RooTable* ttable = data->table(tagCat,"x>8.23") ;
    regTable(ttable,"Basic10_TTable") ;
    
    // Super-category is 'product' of categories
    RooSuperCategory b0Xtcat("b0Xtcat","b0flav X tagCat",RooArgSet(b0flav,tagCat)) ;
    RooTable* bttable = data->table(b0Xtcat) ;
    regTable(bttable,"Basic10_BTTable") ;
    
    // Mapped category is category->category function
    RooMappedCategory tcatType("tcatType","tagCat type",tagCat,"Unknown") ;
    tcatType.map("Lepton","Cut based") ;
    tcatType.map("Kaon","Cut based") ;
    tcatType.map("NetTagger*","Neural Network") ;
    RooTable* mtable = data->table(tcatType) ;
    regTable(mtable,"Basic10_MTable") ;
    
    // Threshold category is real->category function
    RooThresholdCategory xRegion("xRegion","region of x",x,"Background") ;
    xRegion.addThreshold(4.23,"Background") ;
    xRegion.addThreshold(5.23,"SideBand") ;
    xRegion.addThreshold(8.23,"Signal") ;
    xRegion.addThreshold(9.23,"SideBand") ;
    //
    // Background | SideBand | Signal | SideBand | Background
    //           4.23       5.23     8.23       9.23
    data->addColumn(xRegion) ;
    RooPlot* xframe = x.frame() ;
    data->plotOn(xframe) ;
    data->plotOn(xframe,Cut("xRegion==xRegion::SideBand"),MarkerColor(2),MarkerSize(2),Name("Data_Selection")) ;
    regPlot(xframe,"Basic10_Plot1") ;
    
    delete data ;

    return kTRUE ;
  }
} ;
