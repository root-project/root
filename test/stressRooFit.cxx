// @(#)root/roofitcore:$name:  $:$id$
// Authors: Wouter Verkerke  November 2007

#include "TWebFile.h"
#include "TSystem.h"
#include "TString.h"
#include "TStopwatch.h"
#include "TROOT.h"
#include "TLine.h"
#include "TFile.h"
#include "TClass.h"
#include "TCanvas.h"
#include "TBenchmark.h"
#include "RooWorkspace.h"
#include "RooTruthModel.h"
#include "RooTrace.h"
#include "RooThresholdCategory.h"
#include "RooSuperCategory.h"
#include "RooSimultaneous.h"
#include "RooSimPdfBuilder.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooProdPdf.h"
#include "RooPolynomial.h"
#include "RooPlot.h"
#include "RooNumIntConfig.h"
#include "RooNLLVar.h"
#include "RooMsgService.h"
#include "RooMinuit.h"
#include "RooMappedCategory.h"
#include "RooHist.h"
#include "RooGlobalFunc.h"
#include "RooGaussModel.h"
#include "RooGaussian.h"
#include "RooFitResult.h"
#include "RooExtendPdf.h"
#include "RooDouble.h"
#include "RooDecay.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooCurve.h"
#include "RooChi2Var.h"
#include "RooCategory.h"
#include "RooBMixDecay.h"
#include "RooBinning.h"
#include "RooBifurGauss.h"
#include "RooArgusBG.h"
#include "RooAddPdf.h"
#include "RooAddModel.h"
#include "Roo1DTable.h"
#include <string>
#include <list>
#include <iostream>

using namespace std ;
using namespace RooFit ;
   
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
//                                                                           //
// RooFit Examples, Wouter Verkerke                                          //
//                                                                           //
//                                                                           //
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*_*//


Int_t stressRooFit(const char* refFile, Bool_t writeRef, Int_t doVerbose, Int_t oneTest, Bool_t dryRun) ;

static TDirectory* gMemDir = 0 ;

//------------------------------------------------------------------------
void StatusPrint(Int_t id,const TString &title,Int_t status)
{
  // Print test program number and its title
  const Int_t kMAX = 65;
  Char_t number[4];
  sprintf(number,"%2d",id);
  TString header = TString("Test ")+number+" : "+title;
  const Int_t nch = header.Length();
  for (Int_t i = nch; i < kMAX; i++) header += '.';
  cout << header << (status>0 ? "OK" : (status<0 ? "SKIPPED" : "FAILED")) << endl;
}



class RooFitTestUnit : public TObject {
public:
  RooFitTestUnit(TFile* refFile, Bool_t writeRef, Int_t verbose) ;
  ~RooFitTestUnit() ;

  void setSilentMode() ;
  void clearSilentMode() ;
  void regPlot(RooPlot* frame, const char* refName) ;  
  void regResult(RooFitResult* r, const char* refName) ;
  void regValue(Double_t value, const char* refName) ;
  void regTable(RooTable* t, const char* refName) ;
  void regWS(RooWorkspace* ws, const char* refName) ;
  RooWorkspace* getWS(const char* refName) ;
  Bool_t runTest() ;
  Bool_t runCompTests() ;

  virtual Bool_t testCode() = 0 ;  
protected:
  TFile* _refFile ;
  Bool_t _write ;
  Int_t _verb ;
  list<pair<RooPlot*, string> > _regPlots ;
  list<pair<RooFitResult*, string> > _regResults ;
  list<pair<Double_t, string> > _regValues ;
  list<pair<RooTable*,string> > _regTables ;
  list<pair<RooWorkspace*,string> > _regWS ;
} ;


RooFitTestUnit::RooFitTestUnit(TFile* refFile, Bool_t writeRef, Int_t verbose) : _refFile(refFile), _write(writeRef), _verb(verbose)
{
}


RooFitTestUnit::~RooFitTestUnit() 
{
}

void RooFitTestUnit::regPlot(RooPlot* frame, const char* refName) 
{
  if (_refFile) {
    string refNameStr(refName) ;
    _regPlots.push_back(make_pair(frame,refNameStr)) ;
  } else {
    delete frame ;
  }
}

void RooFitTestUnit::regResult(RooFitResult* r, const char* refName) 
{
  if (_refFile) {
    string refNameStr(refName) ;
    _regResults.push_back(make_pair(r,refNameStr)) ;
  } else {
    delete r ;
  }
}

void RooFitTestUnit::regValue(Double_t d, const char* refName) 
{
  if (_refFile) {
    string refNameStr(refName) ;
    _regValues.push_back(make_pair(d,refNameStr)) ;
  }
}

void RooFitTestUnit::regTable(RooTable* t, const char* refName) 
{
  if (_refFile) {
    string refNameStr(refName) ;
    _regTables.push_back(make_pair(t,refNameStr)) ;
  } else {
    delete t ;
  }
}


void RooFitTestUnit::regWS(RooWorkspace* ws, const char* refName) 
{
  if (_refFile) {
    string refNameStr(refName) ;
    _regWS.push_back(make_pair(ws,refNameStr)) ;
  } else {
    delete ws ;
  }
}


RooWorkspace* RooFitTestUnit::getWS(const char* refName) 
{
  RooWorkspace* ws = dynamic_cast<RooWorkspace*>(_refFile->Get(refName)) ;
  if (!ws) {
    cout << "stressRooFit ERROR: cannot retrieve RooWorkspace " << refName 
	 << " from reference file, skipping " << endl ;
    return 0 ;
  }
  
  return ws ;
}


Bool_t RooFitTestUnit::runCompTests() 
{
  Bool_t ret = kTRUE ;

  list<pair<RooPlot*, string> >::iterator iter = _regPlots.begin() ;
  while (iter!=_regPlots.end()) {

    if (!_write) {

      // Comparison mode
      
      // Retrieve benchmark
      RooPlot* bmark = dynamic_cast<RooPlot*>(_refFile->Get(iter->second.c_str())) ;
      if (!bmark) {
	cout << "stressRooFit ERROR: cannot retrieve RooPlot " << iter->second << " from reference file, skipping " << endl ;
	ret = kFALSE ;
	++iter ;
	continue ;
      }
      
      if (_verb) {
	cout << "comparing RooPlot " << iter->first << " to benchmark " << iter->second << " = " << bmark << endl ;      
      }
      
      Stat_t nItems = iter->first->numItems() ;
      for (Stat_t i=0 ; i<nItems ; i++) {
	TObject* obj = iter->first->getObject((Int_t)i) ;
	
	// Retrieve corresponding object from reference frame
	TObject* objRef = bmark->findObject(obj->GetName()) ;
	
	if (!objRef) {
	  cout << "stressRooFit ERROR: cannot retrieve object " << obj->GetName() << " from reference  RooPlot " << iter->second << ", skipping" << endl ;
	  ret = kFALSE ;
	  ++iter ;
	  continue ;
	}
	
	// Histogram comparisons
	if (obj->IsA()==RooHist::Class()) {
	  RooHist* testHist = static_cast<RooHist*>(obj) ;
	  RooHist* refHist = static_cast<RooHist*>(objRef) ;
	  if (!testHist->isIdentical(*refHist)) {
	    cout << "stressRooFit ERROR: comparison of object " << obj->IsA()->GetName() << "::" << obj->GetName() 
		 <<   " fails comparison with counterpart in reference RooPlot " << bmark->GetName() << endl ;
	    ret = kFALSE ;
	  }
	} else if (obj->IsA()==RooCurve::Class()) {
	  RooCurve* testCurve = static_cast<RooCurve*>(obj) ;
	  RooCurve* refCurve = static_cast<RooCurve*>(objRef) ;
	  if (!testCurve->isIdentical(*refCurve)) {
	    cout << "stressRooFit ERROR: comparison of object " << obj->IsA()->GetName() << "::" << obj->GetName() 
		 <<   " fails comparison with counterpart in reference RooPlot " << bmark->GetName() << endl ;
	    ret = kFALSE ;
	  }
	}	
	
      }
      
      // Delete RooPlot when comparison is finished to avoid noise in leak checking
      delete iter->first ;

    } else {

      // Writing mode

      cout <<"stressRooFit: Writing reference RooPlot " << iter->first << " as benchmark " << iter->second << endl ;
      _refFile->cd() ;
      iter->first->Write(iter->second.c_str()) ;
      gMemDir->cd() ;
    }
      
    ++iter ;
  }


  list<pair<RooFitResult*, string> >::iterator iter2 = _regResults.begin() ;
  while (iter2!=_regResults.end()) {

    if (!_write) {

      // Comparison mode
 
     // Retrieve benchmark
      RooFitResult* bmark = dynamic_cast<RooFitResult*>(_refFile->Get(iter2->second.c_str())) ;
      if (!bmark) {
	cout << "stressRooFit ERROR: cannot retrieve RooFitResult " << iter2->second << " from reference file, skipping " << endl ;
	++iter2 ;
	ret = kFALSE ;
	continue ;
      }

      if (_verb) {
	cout << "comparing RooFitResult " << iter2->first << " to benchmark " << iter2->second << " = " << bmark << endl ;      
      }

      if (!iter2->first->isIdentical(*bmark)) {
	cout << "stressRooFit ERROR: comparison of object " << iter2->first->IsA()->GetName() << "::" << iter2->first->GetName() 
	     <<   " fails comparison with counterpart in reference RooFitResult " << bmark->GetName() << endl ;
	ret = kFALSE ;
      }

      // Delete RooFitResult when comparison is finished to avoid noise in leak checking
      delete iter2->first ;
      
        
    } else {

      // Writing mode
      
      cout <<"stressRooFit: Writing reference RooFitResult " << iter2->first << " as benchmark " << iter2->second << endl ;
      _refFile->cd() ;
      iter2->first->Write(iter2->second.c_str()) ;
      gMemDir->cd() ;
    }
      
    ++iter2 ;
  }

  list<pair<Double_t, string> >::iterator iter3 = _regValues.begin() ;
  while (iter3!=_regValues.end()) {

    if (!_write) {

      // Comparison mode
 
     // Retrieve benchmark
      RooDouble* ref = dynamic_cast<RooDouble*>(_refFile->Get(iter3->second.c_str())) ;
      if (!ref) {
	cout << "stressRooFit ERROR: cannot retrieve RooDouble " << iter3->second << " from reference file, skipping " << endl ;
	++iter3 ;
	ret = kFALSE ;
	continue ;
      }
      
      if (_verb) {
	cout << "comparing value " << iter3->first << " to benchmark " << iter3->second << " = " << (Double_t)(*ref) << endl ;      
      }

      if (fabs(iter3->first - (Double_t)(*ref))>1e-6 ) {
	cout << "stressRooFit ERROR: comparison of value " << iter3->first <<   " fails comparison with reference " << ref->GetName() << endl ;
	ret = kFALSE ;
      }
     
        
    } else {

      // Writing mode
      
      cout <<"stressRooFit: Writing reference Double_t " << iter3->first << " as benchmark " << iter3->second << endl ;
      _refFile->cd() ;
      RooDouble* rd = new RooDouble(iter3->first) ;
      rd->Write(iter3->second.c_str()) ;
      gMemDir->cd() ;
    }
      
    ++iter3 ;
  }


  list<pair<RooTable*, string> >::iterator iter4 = _regTables.begin() ;
  while (iter4!=_regTables.end()) {

    if (!_write) {

      // Comparison mode
 
     // Retrieve benchmark
      RooTable* bmark = dynamic_cast<RooTable*>(_refFile->Get(iter4->second.c_str())) ;
      if (!bmark) {
	cout << "stressRooFit ERROR: cannot retrieve RooTable " << iter4->second << " from reference file, skipping " << endl ;
	++iter4 ;
	ret = kFALSE ;
	continue ;
      }

      if (_verb) {
	cout << "comparing RooTable " << iter4->first << " to benchmark " << iter4->second << " = " << bmark << endl ;      
      }

      if (!iter4->first->isIdentical(*bmark)) {
	cout << "stressRooFit ERROR: comparison of object " << iter4->first->IsA()->GetName() << "::" << iter4->first->GetName() 
	     <<   " fails comparison with counterpart in reference RooTable " << bmark->GetName() << endl ;
	ret = kFALSE ;
      }

      // Delete RooTable when comparison is finished to avoid noise in leak checking
      delete iter4->first ;
      
        
    } else {

      // Writing mode
      
      cout <<"stressRooFit: Writing reference RooTable " << iter4->first << " as benchmark " << iter4->second << endl ;
      _refFile->cd() ;
      iter4->first->Write(iter4->second.c_str()) ;
      gMemDir->cd() ;
    }
      
    ++iter4 ;
  }


  list<pair<RooWorkspace*, string> >::iterator iter5 = _regWS.begin() ;
  while (iter5!=_regWS.end()) {

    if (_write) {

      // Writing mode
      
      cout <<"stressRooFit: Writing reference RooWorkspace " << iter5->first << " as benchmark " << iter5->second << endl ;
      _refFile->cd() ;
      iter5->first->Write(iter5->second.c_str()) ;
      gMemDir->cd() ;
    }
      
    ++iter5 ;
  }



  return ret ;
}

void RooFitTestUnit::setSilentMode() 
{
  RooMsgService::instance().setSilentMode(kTRUE) ;
  for (Int_t i=0 ; i<RooMsgService::instance().numStreams() ; i++) {
    if (RooMsgService::instance().getStream(i).minLevel<RooMsgService::ERROR) {
      RooMsgService::instance().setStreamStatus(i,kFALSE) ;
    }
  }
}

void RooFitTestUnit::clearSilentMode() 
{
  RooMsgService::instance().setSilentMode(kFALSE) ;
  for (Int_t i=0 ; i<RooMsgService::instance().numStreams() ; i++) {
    RooMsgService::instance().setStreamStatus(i,kTRUE) ;
  }  
}


Bool_t RooFitTestUnit::runTest()
{
  gMemDir->cd() ;

  if (_verb<2) { 
    setSilentMode() ;
  } else {
    cout << "*** Begin of output of Unit Test at normal verbosity *************" << endl ;
  }  

  RooMsgService::instance().clearErrorCount() ;

  // Reset random generator seed to make results independent of test ordering
  RooRandom::randomGenerator()->SetSeed(12345) ;

  if (!testCode()) return kFALSE ;

  if (_verb<2) { 
    clearSilentMode() ;
  } else {
    cout << "*** End of output of Unit Test at normal verbosity ***************" << endl ;
  }

  if (RooMsgService::instance().errorCount()>0) {
    cout << "RooFitUnitTest: ERROR messages were logged, failing test" << endl ;
    return kFALSE ;
  }

  return runCompTests() ;
}


// ****************   #include "TestBasic1.cxx"


// Elementary operations on a gaussian PDF
class TestBasic1 : public RooFitTestUnit
{
public: 
  TestBasic1(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Generate,Fit and Plot on basic p.d.f"

    // Build Gaussian PDF
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mean("mean","mean of gaussian",-1) ;
    RooRealVar sigma("sigma","width of gaussian",3) ;
    RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;  
        
    // Generate a toy MC set
    RooDataSet* data = gauss.generate(x,10000) ;  
    
    // Fit pdf to toy
    mean.setConstant(kFALSE) ;
    sigma.setConstant(kFALSE) ;
    RooFitResult* r = gauss.fitTo(*data,"mhr") ;
    delete r ;

    // Plot PDF and toy data overlaid
    RooPlot* xframe2 = x.frame() ;
    data->plotOn(xframe2) ;
    gauss.plotOn(xframe2) ;
     
    // Register output frame for comparison test
    // regResult(r,"Basic1_Result") ;
    regPlot(xframe2,"Basic1_Plot") ;

    delete data ;
 
    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic2.cxx"


// Elementary operations on a gaussian PDF
class TestBasic2 : public RooFitTestUnit
{
public: 
  TestBasic2(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Addition operator p.d.f"
    
    // Build two Gaussian PDFs
    RooRealVar x("x","x",0,10) ;
    RooRealVar mean1("mean1","mean of gaussian 1",2,-10,10) ;
    RooRealVar mean2("mean2","mean of gaussian 2",3,-10,10) ;
    RooRealVar sigma("sigma","width of gaussians",1,0.1,10) ;
    RooGaussian gauss1("gauss1","gaussian PDF",x,mean1,sigma) ;  
    RooGaussian gauss2("gauss2","gaussian PDF",x,mean2,sigma) ;  
    
    // Build Argus background PDF
    RooRealVar argpar("argpar","argus shape parameter",-1.,-40.,0.) ;
    RooRealVar cutoff("cutoff","argus cutoff",9.0) ;
    RooArgusBG argus("argus","Argus PDF",x,cutoff,argpar) ;
    
    // Add the components
    RooRealVar g1frac("g1frac","fraction of gauss1",0.5,0,0.7) ;
    RooRealVar g2frac("g2frac","fraction of gauss2",0.3) ;
    RooAddPdf  sum("sum","g1+g2+a",RooArgList(gauss1,gauss2,argus),RooArgList(g1frac,g2frac)) ;
    
    // Generate a toyMC sample
    RooDataSet *data = sum.generate(x,1000) ;

    mean1.setConstant(kTRUE) ;
    mean2.setConstant(kTRUE) ;
    RooFitResult* r = sum.fitTo(*data,"mhr") ;
  
    // Plot data and PDF overlaid
    RooPlot* xframe = x.frame() ;
    data->plotOn(xframe) ;
    
    // Plot only argus and gauss1
    sum.plotOn(xframe) ;
    sum.plotOn(xframe,Components(RooArgSet(argus,gauss2)),Name("curve_Argus_plus_Gauss2")) ;
    sum.plotOn(xframe,Components(argus),Name("curve_Argus")) ;

    regResult(r,"Basic2_Result") ;
    regPlot(xframe,"Basic2_Plot") ;    

    delete data ;

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic3.cxx"


// Elementary operations on a gaussian PDF
class TestBasic3 : public RooFitTestUnit
{
public: 
  TestBasic3(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Product operator p.d.f"

    // Build two Gaussian PDFs
    RooRealVar x("x","x",-5,5) ;
    RooRealVar y("y","y",-5,5) ;
    RooRealVar meanx("mean1","mean of gaussian x",2,-2,6) ;
    RooRealVar meany("mean2","mean of gaussian y",-2) ;
    RooRealVar sigmax("sigmax","width of gaussian x",1.0,0.1,10) ;
    RooRealVar sigmay("sigmay","width of gaussian y",5.0,0.1,50) ;
    RooGaussian gaussx("gaussx","gaussian PDF",x,meanx,sigmax) ;
    RooGaussian gaussy("gaussy","gaussian PDF",y,meany,sigmay) ;
    
    // Multiply the components
    RooProdPdf  prod("gaussxy","gaussx*gaussy",RooArgList(gaussx,gaussy)) ;
    
    // Generate a toyMC sample
    RooDataSet *data = prod.generate(RooArgSet(x,y),1000) ;

    RooFitResult* r = prod.fitTo(*data,"mhr") ;

    RooPlot* xframe = x.frame() ;
    data->plotOn(xframe) ;
    prod.plotOn(xframe) ; // plots f(x) = Int(dy) pdf(x,y)
    
    RooPlot* yframe = y.frame() ;
    data->plotOn(yframe) ;
    prod.plotOn(yframe) ; // plots f(y) = Int(dx) pdf(x,y)

    regResult(r,"Basic3_Result") ;
    regPlot(xframe,"Basic3_PlotX") ;
    regPlot(yframe,"Basic3_PlotY") ;
    
    delete data ;

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic4.cxx"


// Elementary operations on a gaussian PDF
class TestBasic4 : public RooFitTestUnit
{
public: 
  TestBasic4(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Conditional product operator p.d.f"

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic5.cxx"


// Elementary operations on a gaussian PDF
class TestBasic5 : public RooFitTestUnit
{
public: 
  TestBasic5(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Analytically convolved p.d.f"
    
    // Build a simple decay PDF
    RooRealVar dt("dt","dt",-20,20) ;
    RooRealVar tau("tau","tau",1.548) ;
    
    // Build a truth resolution model (delta function)
    RooTruthModel tm("tm","truth model",dt) ;
    
    // Construct a simple unsmeared decay PDF
    RooDecay decay_tm("decay_tm","decay",dt,tau,tm,RooDecay::DoubleSided) ;

    RooPlot* frame1 = dt.frame() ;
    decay_tm.plotOn(frame1) ;
    
    // Build a gaussian resolution model
    RooRealVar bias1("bias1","bias1",0) ;
    RooRealVar sigma1("sigma1","sigma1",1) ;
    RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1) ;
    
    // Construct a decay PDF, smeared with single gaussian resolution model
    RooDecay decay_gm1("decay_gm1","decay",dt,tau,gm1,RooDecay::DoubleSided) ;
    
    RooPlot* frame2 = dt.frame() ;
    decay_gm1.plotOn(frame2) ;
    
    // Build another gaussian resolution model
    RooRealVar bias2("bias2","bias2",0) ;
    RooRealVar sigma2("sigma2","sigma2",5) ;
    RooGaussModel gm2("gm2","gauss model 2",dt,bias2,sigma2) ;
    
    // Build a composite resolution model
    RooRealVar gm1frac("gm1frac","fraction of gm1",0.5) ;
    RooAddModel gmsum("gmsum","sum of gm1 and gm2",RooArgList(gm1,gm2),gm1frac) ;
    
    // Construct a decay PDF, smeared with double gaussian resolution model
    RooDecay decay_gmsum("decay_gmsum","decay",dt,tau,gmsum,RooDecay::DoubleSided) ;
    
    RooPlot* frame3 = dt.frame() ;
    decay_gmsum.plotOn(frame3) ;
    
    regPlot(frame1,"Basic5_Plot1") ;
    regPlot(frame2,"Basic5_Plot2") ;
    regPlot(frame3,"Basic5_Plot3") ;
    

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic6.cxx"


// Elementary operations on a gaussian PDF
class TestBasic6 : public RooFitTestUnit
{
public: 
  TestBasic6(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Simultaneous operator p.d.f"

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic7.cxx"

// Elementary operations on a gaussian PDF
class TestBasic7 : public RooFitTestUnit
{
public: 
  TestBasic7(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {
    
    // "Addition oper. p.d.f with range transformed fractions"

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic8.cxx"


// Elementary operations on a gaussian PDF
class TestBasic8 : public RooFitTestUnit
{
public: 
  TestBasic8(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {
    
    // "Multiple observable configurations of p.d.f.s"

    // A simple gaussian PDF has 3 variables: x,mean,sigma
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mean("mean","mean of gaussian",-1,-10,10) ;
    RooRealVar sigma("sigma","width of gaussian",3,1,20) ;
    RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;
    
    // For getVal() without any arguments all variables are interpreted as parameters,
    // and no normalization is enforced
    x = 0 ;
    Double_t rawVal = gauss.getVal() ; // = exp(-[(x-mean)/sigma]^2)
    regValue(rawVal,"Basic8_Raw") ;
    // cout << "gauss(x=0,mean=-1,width=3)_raw = " << rawVal << endl ;
    
    // If we supply getVal() with the subset of its variables that should be interpreted as dependents,
    // it will apply the correct normalization for that set of dependents
    RooArgSet nset1(x) ;
    Double_t xnormVal = gauss.getVal(&nset1) ; 
    regValue(xnormVal,"Basic8_NormX") ;
    // cout << "gauss(x=0,mean=-1,width=3)_normalized_x[-10,10] = " << xnormVal << endl ;
    
    //*** gauss.getVal(x) = gauss.getVal() / Int(-10,10) gauss() dx
    
    // If we adjust the limits on x, the normalization will change accordingly
    x.setRange(-1,1) ;
    Double_t xnorm2Val = gauss.getVal(&nset1) ;
    regValue(xnorm2Val,"Basic8_NormXRange") ;
    // cout << "gauss(x=0,mean=-1,width=3)_normalized_x[-1,1] = " << xnorm2Val << endl ;
    
    //*** gauss.getVal(x) = gauss.getVal() / Int(-1,1) gauss() dx
    
    // We can also add sigma as dependent
    RooArgSet nset2(x,sigma) ;
    Double_t xsnormVal = gauss.getVal(&nset2) ;
    regValue(xsnormVal,"Basic8_NormXS") ;
    // cout << "gauss(x=0,mean=-1,width=3)_normalized_x[-1,1]_width[1,20] = " << xsnormVal << endl ;
    
    //*** gauss.getVal(RooArgSet(x,sigma)) = gauss.getVal() / Int(-1,1)(1,20) gauss() dx dsigma
    
    
    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic9.cxx"

// Elementary operations on a gaussian PDF
class TestBasic9 : public RooFitTestUnit
{
public: 
  TestBasic9(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Formula expressions"

    // Build Gaussian PDF
    RooRealVar x("x","x",-10,10) ;
    RooRealVar y("y","y",0,3) ;
    
    //  g(x,m,s)
    //  m -> m(y) = m0 + m1*y
    //  g(x,m(y),s)
    
    // Build a parameterized mean variable for gauss
    RooRealVar mean0("mean0","offset of mean function",0.5) ;
    RooRealVar mean1("mean1","slope of mean function",3.0) ;
    RooFormulaVar mean("mean","parameterized mean","mean0+mean1*y",RooArgList(mean0,mean1,y)) ;
    
    RooRealVar sigma("sigma","width of gaussian",3) ;
    RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;
    
    // Generate a toy MC set
    RooDataSet* data = gauss.generate(RooArgList(x,y),10000) ;
        
    // Plot x projection
    RooPlot* xframe = x.frame() ;
    data->plotOn(xframe) ;
    gauss.plotOn(xframe) ; // plots f(x) = Int(dy) pdf(x,y)
    
    // Plot y projection
    RooPlot* yframe = y.frame() ;
    data->plotOn(yframe) ;
    gauss.plotOn(yframe) ; // plots f(y) = Int(dx) pdf(x,y)

    regPlot(xframe,"Basic9_PlotX") ;
    regPlot(yframe,"Basic9_PlotY") ;

    delete data ;

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic10.cxx"


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

// ****************   #include "TestBasic11.cxx"

// Elementary operations on a gaussian PDF
class TestBasic11 : public RooFitTestUnit
{
public: 
  TestBasic11(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

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
      if (!w) return kFALSE ;

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
    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic12.cxx"

// Elementary operations on a gaussian PDF
class TestBasic12 : public RooFitTestUnit
{
public: 
  TestBasic12(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Extended likelihood constructions"


    // Build regular Gaussian PDF
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mean("mean","mean of gaussian",-3,-10,10) ;
    RooRealVar sigma("sigma","width of gaussian",1,0.1,5) ;
    RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;
    
    // Make extended PDF based on gauss. n will be the expected number of events
    RooRealVar n("n","number of events",1000,0,2000) ;
    RooExtendPdf egauss("egauss","extended gaussian PDF",gauss,n) ;
    
    // Generate events from extended PDF
    // The default number of events to generate is taken from gauss.expectedEvents()
    // but can be overrided using a second argument
    RooDataSet* data = egauss.generate(x)  ;
    
    // Fit PDF to dataset in extended mode (selected by fit option "e")
    RooFitResult* r1 = egauss.fitTo(*data,"mher") ;

    
    // Plot both on a frame ;
    RooPlot* xframe = x.frame() ;
    data->plotOn(xframe) ;
    egauss.plotOn(xframe,Normalization(1.0,RooAbsReal::RelativeExpected)) ; // select intrinsic normalization

    // Make an extended gaussian PDF where the number of expected events
    // is counted in a limited region of the dependent range
    x.setRange("cut",-4,2) ;
    RooRealVar mean2("mean2","mean of gaussian",-3) ;
    RooRealVar sigma2("sigma2","width of gaussian",1) ;
    RooGaussian gauss2("gauss2","gaussian PDF 2",x,mean2,sigma2) ;
    
    RooRealVar n2("n2","number of events",1000,0,2000) ;
    RooExtendPdf egauss2("egauss2","extended gaussian PDF w limited range",gauss2,n2,"cut") ;

    RooFitResult* r2 = egauss2.fitTo(*data,"mher");

    
    // cout << "fitted number of events in data in range (-4,2) = " << n2.getVal() << endl ;
    
    // Adding two extended PDFs gives an extended sum PDF
    
    mean = 3.0 ;  sigma = 2.0 ;
    
    // Note that we omit coefficients when adding extended PDFS
    RooAddPdf sumgauss("sumgauss","sum of two extended gauss PDFs",RooArgList(egauss,egauss2)) ;
    sumgauss.plotOn(xframe,LineColor(kRed)) ; // select intrinsic normalization
    
    // Note that in the plot sumgauss does not follow the normalization of the data
    // because its expected number was intentionally chosen not to match the number of events in the data
    
    // If no special 'cut normalizations' are needed (as done in egauss2), there is a shorthand
    // way to construct an extended sumpdf:
    
    RooAddPdf sumgauss2("sumgauss2","extended sum of two gaussian PDFs",
			RooArgList(gauss,gauss2),RooArgList(n,n2)) ;
    sumgauss2.plotOn(xframe,LineColor(kGreen)) ; // select intrinsic normalization
    
    // Note that sumgauss2 looks different from sumgauss because for gauss2 the expected number
    // of event parameter n2 now applies to the entire gauss2 area, whereas in egauss2 it was
    // constructed to represent the number of events in the range (-4,-2). If we would use a separate
    // parameter n3, set to 10000, to represent the number of events for gauss2 in sumgauss2, then
    // sumgauss and sumgauss2 would be indentical.

    regResult(r1,"Basic12_Result1") ;
    regResult(r2,"Basic12_Result2") ;
    regPlot(xframe,"Basic12_Plot1") ;

    delete data ;

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic13.cxx"

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

// ****************   #include "TestBasic14.cxx"

// Elementary operations on a gaussian PDF
class TestBasic14 : public RooFitTestUnit
{
public: 
  TestBasic14(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Plotting with variable bin sizes"

    // Build a simple decay PDF
    RooRealVar dt("dt","dt",-20,20) ;
    RooRealVar dm("dm","dm",0.472) ;
    RooRealVar tau("tau","tau",1.547) ;
    RooRealVar w("w","mistag rate",0.1) ;
    RooRealVar dw("dw","delta mistag rate",0.) ;
    RooCategory mixState("mixState","B0/B0bar mixing state") ;
    mixState.defineType("mixed",-1) ;
    mixState.defineType("unmixed",1) ;
    RooCategory tagFlav("tagFlav","Flavour of the tagged B0") ;
    tagFlav.defineType("B0",1) ;
    tagFlav.defineType("B0bar",-1) ;
    
    // Build a gaussian resolution model
    RooRealVar dterr("dterr","dterr",0.1,1.0) ;
    RooRealVar bias1("bias1","bias1",0) ;
    RooRealVar sigma1("sigma1","sigma1",0.1) ;
    RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1) ;
    
    // Construct a decay PDF, smeared with single gaussian resolution model
    RooBMixDecay bmix("bmix","decay",dt,mixState,tagFlav,tau,dm,w,dw,gm1,RooBMixDecay::DoubleSided) ;
    
    // Generate BMixing data with above set of event errors
    RooDataSet *data = bmix.generate(RooArgSet(dt,mixState,tagFlav),2000) ;
    
    // *** Plot mixState asymmetry with variable bin sizes ***
    
    // Create binning object with range (-10,10)
    RooBinning abins(-10,10) ;
    
    // Define bin boundaries
    abins.addBoundary(0) ;
    abins.addBoundaryPair(1) ;
    abins.addBoundaryPair(2) ;
    abins.addBoundaryPair(3) ;
    abins.addBoundaryPair(4) ;
    abins.addBoundaryPair(6) ;
    
    // Create plot frame and plot mixState asymmetry of bmix
    RooPlot* aframe = dt.frame(-10,10) ;
    bmix.plotOn(aframe,Asymmetry(mixState)) ;
    
    // Plot mixState asymmetry of data with specified binning
    data->plotOn(aframe,Asymmetry(mixState),Binning(abins)) ;
    
    aframe->SetMinimum(-1.1) ;
    aframe->SetMaximum(1.1) ;
    
    // *** Plot deltat distribution with variable bin sizes ***
    
    // Create binning object with range (-15,15)
    RooBinning tbins(-15,15) ;
    
    // Add 60 bins with uniform spacing in range (-15,0)
    tbins.addUniform(60,-15,0) ;
    
    // Make plot with specified binning
    RooPlot* dtframe = dt.frame(-15,15) ;
    data->plotOn(dtframe,Binning(tbins)) ;
    bmix.plotOn(dtframe) ;
    
    regPlot(dtframe,"Basic14_PlotDt") ;
    regPlot(aframe,"Basic14_PlotA") ;

    delete data ;

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic15.cxx"


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

// ****************   #include "TestBasic16.cxx"

// Elementary operations on a gaussian PDF
class TestBasic16 : public RooFitTestUnit
{
public: 
  TestBasic16(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Complex slice projections"

    // Build a simple decay PDF
    RooRealVar dt("dt","dt",-20,20) ;
    RooRealVar dm("dm","dm",0.472) ;
    RooRealVar tau("tau","tau",1.547) ;
    RooRealVar w("w","mistag rate",0.1) ;
    RooRealVar dw("dw","delta mistag rate",0.) ;
    RooCategory mixState("mixState","B0/B0bar mixing state") ;
    mixState.defineType("mixed",-1) ;
    mixState.defineType("unmixed",1) ;
    RooCategory tagFlav("tagFlav","Flavour of the tagged B0") ;
    tagFlav.defineType("B0",1) ;
    tagFlav.defineType("B0bar",-1) ;
    
    // Build a gaussian resolution model
    RooRealVar dterr("dterr","dterr",0.1,1.0) ;
    RooRealVar bias1("bias1","bias1",0) ;
    RooRealVar sigma1("sigma1","sigma1",0.1) ;  
    RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1,dterr) ;
    
    gm1.advertiseFlatScaleFactorIntegral(kTRUE) ; // Pretend that b(X)gm1 is flat in dterr
    
    // Construct a decay PDF, smeared with single gaussian resolution model
    RooBMixDecay bmix("bmix","decay",dt,mixState,tagFlav,tau,dm,w,dw,gm1,RooBMixDecay::DoubleSided) ;
    
    // Define tagCat and runBlock categories
    RooCategory tagCat("tagCat","Tagging category") ;
    tagCat.defineType("Lep") ;
    tagCat.defineType("Kao") ;
    tagCat.defineType("NT1") ;
    tagCat.defineType("NT2") ;
    
    RooCategory runBlock("runBlock","Run block") ;
    runBlock.defineType("Run1") ;
    runBlock.defineType("Run2") ;
    
    // Instantiate a builder
    RooSimPdfBuilder mgr(bmix) ;
    RooArgSet& config = *mgr.createProtoBuildConfig() ;
    
    // Enter configuration data
    config.setStringValue("physModels", "bmix") ;               
    config.setStringValue("splitCats",  "tagCat runBlock") ;    
    config.setStringValue("bmix",       "tagCat          : w " 
			  "runBlock        : tau ") ;
    
    // Build the master PDF
    RooArgSet deps(dt,mixState,dterr,tagCat,runBlock) ;
    RooSimultaneous* simPdf  = (RooSimultaneous*) mgr.buildPdf(config,deps) ;
    RooArgSet& simPars = *simPdf->getParameters(deps) ;
    simPars.setRealValue("tau_Run1",1.5) ;
    simPars.setRealValue("tau_Run2",3.5) ;
    simPars.setRealValue("w_Lep",0.0) ;
    simPars.setRealValue("w_Kao",0.15) ;
    simPars.setRealValue("w_NT1",0.3) ;
    simPars.setRealValue("w_NT2",0.4) ;

    // Generate dummy per-event errors 
    RooBifurGauss gerr("gerr","error distribution",dterr,RooRealConstant::value(0.3),
		       RooRealConstant::value(0.3),RooRealConstant::value(0.8)) ;
    RooDataSet *protoData = gerr.generate(RooArgSet(dterr,tagCat,runBlock),2000) ;      
    
    // Generate other observables
    RooDataSet* data = simPdf->generate(RooArgSet(dt,mixState),*protoData) ;
    
    // Used binned or unbinned projection dataset according to users choice
    RooAbsData* projData = new RooDataHist("projDataWMixH","projDataWMixH",
					   RooArgSet(tagCat,runBlock,dterr,mixState),*data) ;
    
    // Plot the whole dataset
    RooPlot* frame = dt.frame() ;
    data->plotOn(frame) ;
    simPdf->plotOn(frame,ProjWData(dterr,*projData)) ;
    
    // Plot the mixed slice
    RooPlot* frame2 = dt.frame(25) ;
    data->plotOn(frame2,Cut("mixState==mixState::mixed")) ;
    mixState.setLabel("mixed") ;
    simPdf->plotOn(frame2,Slice(mixState),ProjWData(RooArgSet(dterr,mixState),*projData)) ; // Use projData with mixState here
    // Note: if the projected dataset contains value for the sliced observable(s) [as done above], 
    //       only the relevant subset of events is used in the projection 
    //       (i.e events matching the current slice coordinates)
    
    // Plot the {Run1;Lep} slice
    RooPlot* frame3 = dt.frame(25) ;
    data->plotOn(frame3,Cut("runBlock==runBlock::Run1&&tagCat==tagCat::Lep")) ;
    runBlock.setLabel("Run1") ;
    tagCat.setLabel("Lep") ;
    simPdf->plotOn(frame3,Slice(RooArgSet(runBlock,tagCat)),ProjWData(dterr,*projData)) ;
    
    // Plot the {Run1} slice
    RooPlot* frame4 = dt.frame(25) ;
    data->plotOn(frame4,Cut("runBlock==runBlock::Run1")) ;
    runBlock.setLabel("Run1") ;
    simPdf->plotOn(frame4,Slice(runBlock),ProjWData(dterr,*projData)) ;
    
    // Plot the {Run1}-mixed slice
    RooPlot* frame5 = dt.frame(25) ;
    data->plotOn(frame5,Cut("runBlock==runBlock::Run1&&mixState==mixState::mixed")) ;
    runBlock.setLabel("Run1") ;
    mixState.setLabel("mixed") ;
    simPdf->plotOn(frame5,Slice(RooArgSet(runBlock,mixState)),ProjWData(RooArgSet(mixState,dterr),*projData)) ;
    
    // Plot the {Run1;Lep}-mixed slice
    RooPlot* frame6 = dt.frame(25) ;
    data->plotOn(frame6,Cut("runBlock==runBlock::Run1&&tagCat==tagCat::Lep&&mixState==mixState::mixed")) ;
    runBlock.setLabel("Run1") ;
    mixState.setLabel("mixed") ;
    tagCat.setLabel("Lep") ;
    simPdf->plotOn(frame6,Slice(RooArgSet(runBlock,tagCat,mixState)),ProjWData(RooArgSet(dterr,mixState),*projData)) ;

    regPlot(frame, "Basic16_Plot1") ;
    regPlot(frame2,"Basic16_Plot2") ;
    regPlot(frame3,"Basic16_Plot3") ;
    regPlot(frame4,"Basic16_Plot4") ;
    regPlot(frame5,"Basic16_Plot5") ;
    regPlot(frame6,"Basic16_Plot6") ;

    delete &config ;
    delete &simPars ;

    delete protoData ;
    delete data ;
    delete projData ;

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic17.cxx"

// Elementary operations on a gaussian PDF
class TestBasic17 : public RooFitTestUnit
{
public: 
  TestBasic17(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Interactive fitting"

    // Setup a model
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mx("mx","mx",0,-0.5,0.5) ;
    RooRealVar sx("sx","sx",3,2.5,3.5) ;
    RooGaussian gx("gx","gx",x,mx,sx) ;
    
    RooRealVar y("y","y",-10,10) ;
    RooRealVar my("my","my",0,-0.5,0.5) ;
    RooRealVar sy("sy","sy",3,1,10) ;
    RooGaussian gy("gy","gy",y,my,sy) ;
    
    RooProdPdf f("f","f",RooArgSet(gx,gy)) ;
    
    // Generate a toy dataset
    RooDataSet* d = f.generate(RooArgSet(x,y),1000) ;
    
    // Construct likelihood
    RooNLLVar nll("nll","nll",f,*d) ;
    
    // Start Minuit session on nll
    RooMinuit m(nll) ;
    
    // Activate constant-term optimization (always recommended)
    m.optimizeConst(kTRUE) ;
    
    // Run HESSE (mx,my,sx,sy free)
    m.hesse() ;
    RooFitResult* r1 = m.save() ;
    
    // Freeze parameters sx,sy
    sx.setConstant(kTRUE) ;
    sy.setConstant(kTRUE) ;
    // (RooMinuit will fix sx,sy in minuit at the next commmand)
    
    // Run MIGRAD (mx,my free)
    m.migrad() ;


    RooFitResult* r2=m.save() ;
    
    // Release sx
    sx.setConstant(kFALSE) ;
    
    // Run MINOS (mx,my,sx free)
    m.migrad() ;
    m.minos() ;
    
    // Save a snapshot of the fit result
    RooFitResult* r3 = m.save() ;
    
    // Make contour plot of mx vs sx
    // m.contour(mx,my) ;
    
    regResult(r1,"Basic17_Result1") ;
    regResult(r2,"Basic17_Result2") ;
    regResult(r3,"Basic17_Result3") ;

    delete d ;
    
    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic18.cxx"

// Elementary operations on a gaussian PDF
class TestBasic18 : public RooFitTestUnit
{
public: 
  TestBasic18(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Lagrange multipliers"

    // Setup a model
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mx("mx","mx",0,-0.5,0.5) ;
    RooRealVar sx("sx","sx",3,2.5,3.5) ;
    RooGaussian gx("gx","gx",x,mx,sx) ;
    
    RooRealVar y("y","y",-10,10) ;
    RooRealVar my("my","my",0,-0.5,0.5) ;
    RooRealVar sy("sy","sy",3,1,10) ;
    RooGaussian gy("gy","gy",y,my,sy) ;
    
    RooProdPdf f("f","f",RooArgSet(gx,gy)) ;
    
    // Generate a toy dataset
    RooDataSet* d = f.generate(RooArgSet(x,y),1000) ;
    
    // Construct likelihood
    RooNLLVar nll("nll","nll",f,*d) ;
    
    // Construct formula with likelihood plus penalty
    RooRealVar alpha("alpha","penalty strength",1000) ;
    RooFormulaVar nllPen("nllPen",
			 "nll+alpha*abs(mx-my)",
			 RooArgList(nll,alpha,mx,my)) ;
    
    // Start Minuit session on straight NLL
    RooMinuit m(nll) ;
    m.migrad() ;
    m.hesse() ;
    RooFitResult* r1 = m.save() ;
    
    // Start Minuit session on straight NLL
    RooMinuit m2(nllPen) ;
    m2.migrad() ;
    m2.hesse() ;
    RooFitResult* r2 = m2.save() ;
    
    // Print results
    regResult(r1,"Basic18_Result1") ;
    regResult(r2,"Basic18_Result2") ;
    
    delete d ;

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic19.cxx"

// Elementary operations on a gaussian PDF
class TestBasic19 : public RooFitTestUnit
{
public: 
  TestBasic19(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Chi^2 fits"

    // Setup a model
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mx("mx","mx",0,-0.5,0.5) ;
    RooRealVar sx("sx","sx",3,2.5,3.5) ;
    RooGaussian gx("gx","gx",x,mx,sx) ;
    
    RooRealVar y("y","y",-10,10) ;
    RooRealVar my("my","my",0,-0.5,0.5) ;
    RooRealVar sy("sy","sy",3,1,10) ;
    RooGaussian gy("gy","gy",y,my,sy) ;
    
    RooProdPdf f("f","f",RooArgSet(gx,gy)) ;
    
    // Generate a toy dataset
    RooDataSet* d = f.generate(RooArgSet(x,y),10000) ;
    
    // Bin dataset
    x.setBins(15) ;
    y.setBins(15) ;
    RooDataHist* db = new RooDataHist("db","db",RooArgSet(x,y),*d) ;
    
    // Construct binned likelihood
    RooNLLVar nll("nll","nll",f,*db) ;
    
    // Start Minuit session on NLL
    RooMinuit m(nll) ;
    m.migrad() ;
    m.hesse() ;
    RooFitResult* r1 = m.save() ;
    
    // Construct Chi2
    RooChi2Var chi2("chi2","chi2",f,*db) ;
    
    // Start Minuit session on Chi2
    RooMinuit m2(chi2) ;
    m2.migrad() ;
    m2.hesse() ;
    RooFitResult* r2 = m2.save() ;
    
    // Print results
    regResult(r1,"Basic19_ResultBLL") ;
    regResult(r2,"Basic19_ResultChi2") ;

    delete db ;
    delete d ;
    
    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic20.cxx"

// Elementary operations on a gaussian PDF
class TestBasic20 : public RooFitTestUnit
{
public: 
  TestBasic20(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {
    
    // "Use of conditional observables" 

    // Build a simple decay PDF
    RooRealVar dt("dt","dt",-20,20) ;
    RooRealVar dm("dm","dm",0.472,0.1,1.0) ;
    RooRealVar tau("tau","tau",1.548,0.1,5.0) ;
    RooRealVar w("w","mistag rate",0.1,0.0,0.5) ;
    RooRealVar dw("dw","delta mistag rate",0.) ;
    RooCategory mixState("mixState","B0/B0bar mixing state") ;
    mixState.defineType("mixed",-1) ;
    mixState.defineType("unmixed",1) ;
    RooCategory tagFlav("tagFlav","Flavour of the tagged B0") ;
    tagFlav.defineType("B0",1) ;
    tagFlav.defineType("B0bar",-1) ;
    
    // Build a gaussian resolution model
    RooRealVar dterr("dterr","dterr",0.1,5.0) ;
    RooRealVar bias1("bias1","bias1",0) ;
    RooRealVar sigma1("sigma1","sigma1",1) ;
    
    RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1,dterr) ;
    
    // Construct a decay PDF, smeared with single gaussian resolution model
    RooBMixDecay bmix("bmix","decay",dt,mixState,tagFlav,tau,dm,w,dw,gm1,RooBMixDecay::DoubleSided) ;
    
    // Generate a set of event errors
    RooBifurGauss gerr("gerr","error distribution",dterr,RooRealConstant::value(0.3),
		       RooRealConstant::value(0.3),RooRealConstant::value(0.8)) ;
    RooDataSet *errdata = gerr.generate(dterr,500) ;
    
    // Generate BMixing data with above set of event errors
    RooDataSet *data = bmix.generate(RooArgSet(dt,mixState,tagFlav),*errdata,500) ;
    
    RooFitResult* r = bmix.fitTo(*data,ConditionalObservables(dterr),FitOptions("mhr")) ;

    
    RooPlot* dtframe1 = dt.frame() ;
    data->plotOn(dtframe1) ;
    bmix.plotOn(dtframe1,ProjWData(dterr,*data)) ; // automatically projects (sums) over mixState
    
    RooPlot* dtframe2 = dt.frame() ;
    data->plotOn(dtframe2,Cut("mixState==mixState::mixed")) ;
    mixState.setLabel("mixed") ;
    bmix.plotOn(dtframe2,Slice(mixState),ProjWData(RooArgSet(mixState,dterr),*data)) ; // projects slice in mixState with mixState=mixed
    
    RooPlot* dtframe3 = dt.frame() ;
    data->plotOn(dtframe3,Cut("mixState==mixState::unmixed")) ;
    mixState.setLabel("unmixed") ;
    bmix.plotOn(dtframe3,Slice(mixState),ProjWData(RooArgSet(mixState,dterr),*data)) ;// projects slice in mixState with mixState=mixed

    regResult(r,"Basic20_Result1") ;
    regPlot(dtframe1,"Basic20_Plot1") ;
    regPlot(dtframe2,"Basic20_Plot2") ;
    regPlot(dtframe3,"Basic20_Plot3") ;

    delete data ;
    delete errdata ;
    
    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic21.cxx"


// Elementary operations on a gaussian PDF
class TestBasic21 : public RooFitTestUnit
{
public: 
  TestBasic21(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Constant term optimization"

    return kTRUE ;
  }
} ;

// ****************   #include "TestBasic22.cxx"

// Elementary operations on a gaussian PDF
class TestBasic22 : public RooFitTestUnit
{
public: 
  TestBasic22(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Numeric Integration"

    return kTRUE ;
  }
} ;


//______________________________________________________________________________
Int_t stressRooFit(const char* refFile, Bool_t writeRef, Int_t doVerbose, Int_t oneTest, Bool_t dryRun)
{
  // Save memory directory location
  gMemDir = gDirectory ;

  TFile* fref = 0 ;
  if (!dryRun) {
    if (TString(refFile).Contains("http:")) {
      if (writeRef) {
	cout << "stressRooFit ERROR: reference file must be local file in writing mode" << endl ;
	return kFALSE ;
      }
      fref = new TWebFile(refFile) ;
    } else {
      fref = new TFile(refFile,writeRef?"RECREATE":"") ;
    }
    if (fref->IsZombie()) {
      cout << "stressRooFit ERROR: cannot open reference file " << refFile << endl ;
      return kFALSE ;
    }
  }

  if (dryRun) {
    // Preload singletons here so they don't show up in trace accounting
    RooNumIntConfig::defaultConfig() ;
    RooResolutionModel::identity() ;

    RooTrace::active(1) ;
  }

  // Add dedicated logging stream for errors that will remain active in silent mode
  RooMsgService::instance().addStream(RooMsgService::ERROR) ;

  cout << "******************************************************************" <<endl;
  cout << "*  RooFit - S T R E S S suite                                    *" <<endl;
  cout << "******************************************************************" <<endl;
  cout << "******************************************************************" <<endl;
  
  TStopwatch timer;
  timer.Start();
  
  cout << "*  Starting  S T R E S S  basic suite                            *" <<endl;
  cout << "******************************************************************" <<endl;
  
  gBenchmark->Start("StressRooFit");

  if (oneTest<0 || oneTest==1) {
    TestBasic1 test1(fref,writeRef,doVerbose) ;
    StatusPrint( 1,"Generate,Fit and Plot on basic p.d.f",test1.runTest());
  }

  if (oneTest<0 || oneTest==2) {
    TestBasic2 test2(fref,writeRef,doVerbose) ;
    StatusPrint( 2,"Addition operator p.d.f",test2.runTest());
  }

  if (oneTest<0 || oneTest==3) {
    TestBasic3 test3(fref,writeRef,doVerbose) ;
    StatusPrint( 3,"Product operator p.d.f",test3.runTest());
  }

  if (oneTest<0 || oneTest==4) {
    TestBasic4 test4(fref,writeRef,doVerbose) ;
    StatusPrint( 4,"Conditional product operator p.d.f",01); // to do
  }

  if (oneTest<0 || oneTest==5) {
    TestBasic5 test5(fref,writeRef,doVerbose) ;
    StatusPrint( 5,"Analytically convolved p.d.f",test5.runTest());
  }

  if (oneTest<0 || oneTest==6) {
    TestBasic6 test6(fref,writeRef,doVerbose) ;
    StatusPrint( 6,"Simultaneous operator p.d.f",-1); // to do
  }

  if (oneTest<0 || oneTest==7) {
    TestBasic7 test7(fref,writeRef,doVerbose) ;
    StatusPrint( 7,"Addition oper. p.d.f with range transformed fractions",-1); // to do
  }

  if (oneTest<0 || oneTest==8) {
    TestBasic8 test8(fref,writeRef,doVerbose) ;
    StatusPrint( 8,"Multiple observable configurations of p.d.f.s",test8.runTest()) ;
  }

  if (oneTest<0 || oneTest==9) {
    TestBasic9 test9(fref,writeRef,doVerbose) ;
    StatusPrint( 9,"Formula expressions",test9.runTest()) ;
  }

  if (oneTest<0 || oneTest==10) {
    TestBasic10 test10(fref,writeRef,doVerbose) ;
    StatusPrint(10,"Functions of discrete variables",test10.runTest()) ;
  }

  if (oneTest<0 || oneTest==11) {
    TestBasic11 test11(fref,writeRef,doVerbose) ;
    StatusPrint(11,"Workspace persistence",test11.runTest()) ; 
  }

  if (oneTest<0 || oneTest==12) {
    TestBasic12 test12(fref,writeRef,doVerbose) ;
    StatusPrint(12,"Extended likelihood constructions",test12.runTest()) ;
  }

  if (oneTest<0 || oneTest==13) {
    TestBasic13 test13(fref,writeRef,doVerbose) ;
    StatusPrint(13,"Data import and persistence",test13.runTest()) ;
  }

  if (oneTest<0 || oneTest==14) {
    TestBasic14 test14(fref,writeRef,doVerbose) ;
    StatusPrint(14,"Plotting with variable bin sizes",test14.runTest()) ;
  }

  if (oneTest<0 || oneTest==15) {
    TestBasic15 test15(fref,writeRef,doVerbose) ;
    StatusPrint(15,"Likelihood ratio projections",test15.runTest()) ;
  }

  if (oneTest<0 || oneTest==16) {
    TestBasic16 test16(fref,writeRef,doVerbose) ;
    StatusPrint(16,"Complex slice projections",test16.runTest()) ;
  }

  if (oneTest<0 || oneTest==17) {
    TestBasic17 test17(fref,writeRef,doVerbose) ;
    StatusPrint(17,"Interactive fitting",test17.runTest()) ;
  }

  if (oneTest<0 || oneTest==18) {
    TestBasic18 test18(fref,writeRef,doVerbose) ;
    StatusPrint(18,"Lagrange multipliers",test18.runTest()) ;
  }

  if (oneTest<0 || oneTest==19) {
    TestBasic19 test19(fref,writeRef,doVerbose) ;
    StatusPrint(19,"Chi^2 fits",test19.runTest()) ;
  }

  if (oneTest<0 || oneTest==20) {
    TestBasic20 test20(fref,writeRef,doVerbose) ;
    StatusPrint(20,"Use of conditional observables",test20.runTest()) ;
  }

  if (oneTest<0 || oneTest==21) {
    TestBasic21 test21(fref,writeRef,doVerbose) ;
    StatusPrint(21,"Constant term optimization",-1) ; // to do
  }

  if (oneTest<0 || oneTest==22) {
    TestBasic22 test22(fref,writeRef,doVerbose) ;
    StatusPrint(22,"Numeric Integration",-1) ; // to do
  }

  if (dryRun) {
    RooTrace::dump() ;
  }

  gBenchmark->Stop("StressRooFit");
  
  
  //Print table with results
  Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
  printf("******************************************************************\n");
  if (UNIX) {
    FILE *fp = gSystem->OpenPipe("uname -a", "r");
    Char_t line[60];
    fgets(line,60,fp); line[59] = 0;
    printf("*  SYS: %s\n",line);
    gSystem->ClosePipe(fp);
  } else {
    const Char_t *os = gSystem->Getenv("OS");
    if (!os) printf("*  SYS: Windows 95\n");
    else     printf("*  SYS: %s %s \n",os,gSystem->Getenv("PROCESSOR_IDENTIFIER"));
  }
  
  printf("******************************************************************\n");
  gBenchmark->Print("StressFit");
#ifdef __CINT__
  Double_t reftime = 86.34; //pcbrun4 interpreted
#else
  Double_t reftime = 21.27; //pcbrun4 compiled
#endif
  const Double_t rootmarks = 860*reftime/gBenchmark->GetCpuTime("StressRooFit");
  
  printf("******************************************************************\n");
  printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),
         gROOT->GetVersionDate(),gROOT->GetVersionTime());
  printf("******************************************************************\n");
  
  printf("Time at the end of job = %f seconds\n",timer.CpuTime());

  if (fref) {
    fref->Close() ;
    delete fref ;
  }

  delete gBenchmark ;
  gBenchmark = 0 ;

  return 0;
}

//_____________________________batch only_____________________
#ifndef __CINT__

int main(int argc,const char *argv[]) 
{
  Bool_t doWrite = kFALSE ;
  Int_t doVerbose = 0 ;
  Int_t oneTest   = -1 ;
  Int_t dryRun    = kFALSE ;

  string refFileName = "http://root.cern.ch/files/stressRooFit_ref.root" ;
  if (gROOT->GetVersionInt() > 51900)
     refFileName = "http://root.cern.ch/files/stressRooFit_ref_2.root" ;
  // Parse command line arguments 
  for (Int_t i=1 ;  i<argc ; i++) {
    string arg = argv[i] ;

    if (arg=="-r") {
      cout << "stressRooFit: using reference file " << argv[i+1] << endl ;
      refFileName = argv[++i] ;
    }

    if (arg=="-w") {
      cout << "stressRooFit: running in writing mode to updating reference file" << endl ;
      doWrite = kTRUE ;
    }

    if (arg=="-mc") {
      cout << "stressRooFit: running in memcheck mode, no regression tests are performed" << endl ;
      dryRun=kTRUE ;
    }

    if (arg=="-v") {
      cout << "stressRooFit: running in verbose mode" << endl ;
      doVerbose = 1 ;
    }

    if (arg=="-vv") {
      cout << "stressRooFit: running in very verbose mode" << endl ;
      doVerbose = 2 ;
    }

    if (arg=="-n") {
      cout << "stressRooFit: running single test " << argv[i+1] << endl ;
      oneTest = atoi(argv[++i]) ;      
    }
    
    if (arg=="-d") {
      cout << "stressRooFit: setting gDebug to " << argv[i+1] << endl ;
      gDebug = atoi(argv[++i]) ;
    }

   }

  if (doWrite && refFileName.find("http:")==0) {

    // Locate file name part in URL and update refFileName accordingly
    char* buf = new char[refFileName.size()+1] ;
    strcpy(buf,refFileName.c_str()) ;
    char *ptr = strrchr(buf,'/') ;
    if (!ptr) {
      ptr = strrchr(buf,':') ;
    }
    refFileName = ptr+1 ;
    delete[] buf ;

    cout << "stressRooFit: WARNING running in write mode, but reference file is web file, writing local file instead: " << refFileName << endl ;
  }

  gBenchmark = new TBenchmark();
  stressRooFit(refFileName.c_str(),doWrite,doVerbose,oneTest,dryRun);  
  return 0;
}

#endif
