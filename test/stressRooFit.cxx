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
#include "TH1.h"
#include "TBenchmark.h"
#include "RooGlobalFunc.h"
#include "RooMsgService.h"
#include "RooPlot.h"
#include "RooFitResult.h"
#include "RooDouble.h"
#include "RooWorkspace.h"
#include "Roo1DTable.h"
#include "RooCurve.h"
#include "RooHist.h"
#include "RooRandom.h"
#include "RooTrace.h"
#include "RooMath.h"
#include <string>
#include <list>
#include <iostream>
#include <math.h>

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



class RooFitTestUnit : public TNamed {
public:
  RooFitTestUnit(const char* name, TFile* refFile, Bool_t writeRef, Int_t verbose) ;
  ~RooFitTestUnit() ;
  
  void setDebug(Bool_t flag) { _debug = flag ; }
  void setSilentMode() ;
  void clearSilentMode() ;
  void regPlot(RooPlot* frame, const char* refName) ;  
  void regResult(RooFitResult* r, const char* refName) ;
  void regValue(Double_t value, const char* refName) ;
  void regTable(RooTable* t, const char* refName) ;
  void regWS(RooWorkspace* ws, const char* refName) ;
  void regTH(TH1* h, const char* refName) ;
  RooWorkspace* getWS(const char* refName) ;
  Bool_t runTest() ;
  Bool_t runCompTests() ;
  Bool_t areTHidentical(TH1* htest, TH1* href) ;

  virtual Bool_t isTestAvailable() { return kTRUE ; }
  virtual Bool_t testCode() = 0 ;  

  virtual Double_t htol() { return 5e-4 ; } // histogram test tolerance (KS dist != prob)
  virtual Double_t ctol() { return 2e-3 ; } // curve test tolerance
  virtual Double_t fptol() { return 1e-3 ; } // fit parameter test tolerance
  virtual Double_t fctol() { return 1e-3 ; } // fit correlation test tolerance
  virtual Double_t vtol() { return 1e-3 ; } // value test tolerance

protected:
  TFile* _refFile ;
  Bool_t _debug ;
  Bool_t _write ;
  Int_t _verb ;
  list<pair<RooPlot*, string> > _regPlots ;
  list<pair<RooFitResult*, string> > _regResults ;
  list<pair<Double_t, string> > _regValues ;
  list<pair<RooTable*,string> > _regTables ;
  list<pair<RooWorkspace*,string> > _regWS ;
  list<pair<TH1*,string> > _regTH ;
} ;


RooFitTestUnit::RooFitTestUnit(const char* name, TFile* refFile, Bool_t writeRef, Int_t verbose) : TNamed(name,name), 
  			                         _refFile(refFile), _debug(kFALSE), _write(writeRef), _verb(verbose)
{
}


RooFitTestUnit::~RooFitTestUnit() 
{
}

void RooFitTestUnit::regPlot(RooPlot* frame, const char* refName) 
{
  if (_refFile) {
    string refNameStr(refName) ;
    frame->SetName(refName) ;
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


void RooFitTestUnit::regTH(TH1* th, const char* refName) 
{
  if (_refFile) {
    string refNameStr(refName) ;
    _regTH.push_back(make_pair(th,refNameStr)) ;
  } else {
    delete th ;
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


Bool_t RooFitTestUnit::areTHidentical(TH1* htest, TH1* href) 
{
  if (htest->GetDimension() != href->GetDimension()) {
    return kFALSE ;
  }

  // Use Kolmogorov distance as metric rather than probability
  // because we expect histograms to be identical rather
  // than drawn from the same parent distribution
  Double_t kmax = htest->KolmogorovTest(href,"M") ;

  if (kmax>htol()) {

    cout << "KS distances = " << kmax << endl ;

    Int_t ntest = htest->GetNbinsX() +2 ;
    Int_t nref  = href->GetNbinsX() +2 ;
    if (htest->GetDimension()>1) {
      ntest *= htest->GetNbinsY() + 2 ;
      nref *= href->GetNbinsY() + 2 ;
    }
    if (htest->GetDimension()>2) {
      ntest *= htest->GetNbinsZ() + 2 ;
      nref *= href->GetNbinsZ() + 2 ;
    }
    
    if (ntest != nref) {
      return kFALSE ;
    }
    
    for (Int_t i=0 ; i<ntest ; i++) {
      if (fabs(htest->GetBinContent(i)-href->GetBinContent(i))>htol()) {
	cout << "htest[" << i << "] = " << htest->GetBinContent(i) << " href[" << i << "] = " << href->GetBinContent(i) << endl; 
      }
    }

    return kFALSE ;
  }

  return kTRUE ;
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
	cout << "reference: " ; iter->first->Print() ;
	cout << "benchmark: " ; bmark->Print() ;
      }

      RooPlot* compPlot = _debug ? iter->first->emptyClone(Form("%s_comparison",iter->first->GetName())) : 0 ;
      Bool_t anyFail=kFALSE ;
      
      Stat_t nItems = iter->first->numItems() ;
      for (Stat_t i=0 ; i<nItems ; i++) {
	TObject* obj = iter->first->getObject((Int_t)i) ;
	
	// Retrieve corresponding object from reference frame
	TObject* objRef = bmark->findObject(obj->GetName()) ;
	
	if (!objRef) {
	  cout << "stressRooFit ERROR: cannot retrieve object " << obj->GetName() << " from reference  RooPlot " << iter->second << ", skipping" << endl ;
	  ret = kFALSE ;
	  break ;
	}
	
	// Histogram comparisons
	if (obj->IsA()==RooHist::Class()) {
	  RooHist* testHist = static_cast<RooHist*>(obj) ;
	  RooHist* refHist = static_cast<RooHist*>(objRef) ;
	  if (!testHist->isIdentical(*refHist,htol())) {
	    cout << "stressRooFit ERROR: comparison of object " << obj->IsA()->GetName() << "::" << obj->GetName() 
		 <<   " fails comparison with counterpart in reference RooPlot " << bmark->GetName() << endl ;
	    
	    if (compPlot) {
	      compPlot->addPlotable((RooHist*)testHist->Clone(),"P") ;	    
	      compPlot->getAttLine()->SetLineColor(kRed) ;
	      compPlot->getAttMarker()->SetMarkerColor(kRed) ;
	      compPlot->getAttLine()->SetLineWidth(1) ;
	      
	      compPlot->addPlotable((RooHist*)refHist->Clone(),"P") ;
	      compPlot->getAttLine()->SetLineColor(kBlue) ;
	      compPlot->getAttMarker()->SetMarkerColor(kBlue) ;
	      compPlot->getAttLine()->SetLineWidth(1) ;
	    }

	    anyFail=kTRUE ;
	    ret = kFALSE ;
	  }
	} else if (obj->IsA()==RooCurve::Class()) {
	  RooCurve* testCurve = static_cast<RooCurve*>(obj) ;
	  RooCurve* refCurve = static_cast<RooCurve*>(objRef) ;
	  if (!testCurve->isIdentical(*refCurve,ctol())) {
	    cout << "stressRooFit ERROR: comparison of object " << obj->IsA()->GetName() << "::" << obj->GetName() 
		 <<   " fails comparison with counterpart in reference RooPlot " << bmark->GetName() << endl ;

	    if (compPlot) {
	      compPlot->addPlotable((RooCurve*)testCurve->Clone()) ;
	      compPlot->getAttLine()->SetLineColor(kRed) ;
	      compPlot->getAttLine()->SetLineWidth(1) ;
	      compPlot->getAttLine()->SetLineStyle(kSolid) ;
	      
	      compPlot->addPlotable((RooCurve*)refCurve->Clone()) ;
	      compPlot->getAttLine()->SetLineColor(kBlue) ;
	      compPlot->getAttLine()->SetLineWidth(1) ;
	      compPlot->getAttLine()->SetLineStyle(kDashed) ;
	    }

	    anyFail=kTRUE ;
	    ret = kFALSE ;
	  }
	  
	}	
	
      }

      if (anyFail && compPlot) {
	cout << "stressRooFit INFO: writing comparison plot " << compPlot->GetName() << " of failed test to stressRooFit_DEBUG.root" << endl ;
	TFile fdbg("stressRooFit_DEBUG.root","UPDATE") ;
	compPlot->Write() ;
	fdbg.Close() ;
      } else {
	delete compPlot ;
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

      if (!iter2->first->isIdentical(*bmark,fptol(),fctol())) {
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

      if (fabs(iter3->first - (Double_t)(*ref))>vtol() ) {
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

  /////////////////
  list<pair<TH1*, string> >::iterator iter6 = _regTH.begin() ;
  while (iter6!=_regTH.end()) {

    if (!_write) {

      // Comparison mode
 
     // Retrieve benchmark
      TH1* bmark = dynamic_cast<TH1*>(_refFile->Get(iter6->second.c_str())) ;
      if (!bmark) {
	cout << "stressRooFit ERROR: cannot retrieve TH1 " << iter6->second << " from reference file, skipping " << endl ;
	++iter6 ;
	ret = kFALSE ;
	continue ;
      }

      if (_verb) {
	cout << "comparing TH1 " << iter6->first << " to benchmark " << iter6->second << " = " << bmark << endl ;      
      }

      if (!areTHidentical(iter6->first,bmark)) {
	cout << "stressRooFit ERROR: comparison of object " << iter6->first->IsA()->GetName() << "::" << iter6->first->GetName() 
	     <<   " fails comparison with counterpart in reference TH1 " << bmark->GetName() << endl ;


      if (_debug) {
	cout << "stressRooFit INFO: writing THx " << iter6->first->GetName() << " and " << bmark->GetName()  
	     << " of failed test to stressRooFit_DEBUG.root" << endl ;
	TFile fdbg("stressRooFit_DEBUG.root","UPDATE") ;
	iter6->first->SetName(Form("%s_test",iter6->first->GetName())) ;
	iter6->first->Write() ;
	bmark->SetName(Form("%s_ref",bmark->GetName())) ;
	bmark->Write() ;
	fdbg.Close() ;
      }

	ret = kFALSE ;
      }

      // Delete TH1 when comparison is finished to avoid noise in leak checking
      delete iter6->first ;
      
        
    } else {

      // Writing mode
      
      cout <<"stressRooFit: Writing reference TH1 " << iter6->first << " as benchmark " << iter6->second << endl ;
      _refFile->cd() ;
      iter6->first->Write(iter6->second.c_str()) ;
      gMemDir->cd() ;
    }
      
    ++iter6 ;
  }


  /////////////////

  return ret ;
}

void RooFitTestUnit::setSilentMode() 
{
  RooMsgService::instance().setSilentMode(kTRUE) ;
  for (Int_t i=0 ; i<RooMsgService::instance().numStreams() ; i++) {
    if (RooMsgService::instance().getStream(i).minLevel<RooFit::ERROR) {
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
  gRandom->SetSeed(12345) ;
  RooRandom::randomGenerator()->SetSeed(12345) ;

  RooTrace::callgrind_zero() ;
  if (!testCode()) return kFALSE ;
  RooTrace::callgrind_dump() ;

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


//#include "stressRooFit_tests_direct.cxx"
#include "stressRooFit_tests.cxx"

//______________________________________________________________________________
Int_t stressRooFit(const char* refFile, Bool_t writeRef, Int_t doVerbose, Int_t oneTest, Bool_t dryRun, Bool_t doDump, Bool_t doVectorStore)
{
  // Save memory directory location
  gMemDir = gDirectory ;

  if (doVectorStore) {
    RooAbsData::defaultStorageType=RooAbsData::Vector ;
  }

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
  RooMsgService::instance().addStream(RooFit::ERROR) ;

  cout << "******************************************************************" <<endl;
  cout << "*  RooFit - S T R E S S suite                                    *" <<endl;
  cout << "******************************************************************" <<endl;
  cout << "******************************************************************" <<endl;
  
  TStopwatch timer;
  timer.Start();

  list<RooFitTestUnit*> testList ;
  testList.push_back(new TestBasic101(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic102(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic103(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic105(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic108(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic109(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic110(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic111(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic201(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic202(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic203(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic204(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic205(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic208(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic209(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic301(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic302(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic303(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic304(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic305(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic306(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic307(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic308(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic310(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic311(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic312(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic313(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic315(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic314(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic316(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic402(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic403(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic404(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic405(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic406(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic501(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic599(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic601(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic602(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic604(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic605(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic606(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic607(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic609(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic701(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic702(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic703(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic704(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic705(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic706(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic707(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic708(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic801(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic802(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic803(fref,writeRef,doVerbose)) ;
  testList.push_back(new TestBasic804(fref,writeRef,doVerbose)) ;
  
  cout << "*  Starting  S T R E S S  basic suite                            *" <<endl;
  cout << "******************************************************************" <<endl;

  if (doDump) {
    TFile fdbg("stressRooFit_DEBUG.root","RECREATE") ;
  }  
  
  gBenchmark->Start("StressRooFit");

  Int_t i(1) ;
  for (list<RooFitTestUnit*>::iterator iter = testList.begin () ; iter != testList.end() ; ++iter) {
    if (oneTest<0 || oneTest==i) {
      if (doDump) {
	(*iter)->setDebug(kTRUE) ;
      }
      StatusPrint( i,(*iter)->GetName(),(*iter)->isTestAvailable()?(*iter)->runTest():-1);
    }
    delete (*iter) ;
    i++ ;
  }

  if (dryRun) {
    RooTrace::dump() ;
  }

  gBenchmark->Stop("StressRooFit");
  
  
  //Print table with results
  Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
  printf("******************************************************************\n");
  if (UNIX) {
     TString sp = gSystem->GetFromPipe("uname -a");
     sp.Resize(60);
     printf("*  SYS: %s\n",sp.Data());
     if (strstr(gSystem->GetBuildNode(),"Linux")) {
        sp = gSystem->GetFromPipe("lsb_release -d -s");
        printf("*  SYS: %s\n",sp.Data());
     }
     if (strstr(gSystem->GetBuildNode(),"Darwin")) {
        sp  = gSystem->GetFromPipe("sw_vers -productVersion");
        sp += " Mac OS X ";
        printf("*  SYS: %s\n",sp.Data());
     }
  } else {
    const Char_t *os = gSystem->Getenv("OS");
    if (!os) printf("*  SYS: Windows 95\n");
    else     printf("*  SYS: %s %s \n",os,gSystem->Getenv("PROCESSOR_IDENTIFIER"));
  }
  
  printf("******************************************************************\n");
  gBenchmark->Print("StressFit");
#ifdef __CINT__
  Double_t reftime = 186.34; //pcbrun4 interpreted
#else
  Double_t reftime = 93.59; //pcbrun4 compiled
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
  Bool_t doWrite     = kFALSE ;
  Int_t doVerbose    = 0 ;
  Int_t oneTest      = -1 ;
  Int_t dryRun       = kFALSE ;
  Bool_t doDump      = kFALSE ;
  Bool_t doVectorStore = kFALSE ;

  string refFileName = "http://root.cern.ch/files/stressRooFit_v530_ref.root" ;

  // Parse command line arguments 
  for (Int_t i=1 ;  i<argc ; i++) {
    string arg = argv[i] ;

    if (arg=="-f") {
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

    if (arg=="-vs") {
      cout << "stressRooFit: setting vector-based storage for datasets" << endl ;
      doVectorStore=kTRUE ;
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
    
    if (arg=="-c") {
      cout << "stressRooFit: dumping comparison file for failed tests " << endl ;
      doDump=kTRUE ;
    }

    if (arg=="-h") {
      cout << "usage: stressRooFit [ options ] " << endl ;
      cout << "" << endl ;
      cout << "       -f <file> : use given reference file instead of default (" <<  refFileName << ")" << endl ;
      cout << "       -w        : write reference file, instead of reading file and running comparison tests" << endl ;
      cout << " " << endl ;
      cout << "       -n N      : Only run test with sequential number N instead of full suite of tests" << endl ;
      cout << "       -c        : dump file stressRooFit_DEBUG.root to which results of both current result and reference for each failed test are written" << endl ;
      cout << "       -mc       : memory check mode, no regression test are performed. Set this flag when running with valgrind" << endl ;
      cout << "       -vs       : Use vector-based storage for all datasets (default is tree-based storage)" << endl ;
      cout << "       -v/-vv    : set verbose mode (show result of each regression test) or very verbose mode (show all roofit output as well)" << endl ;
      cout << "       -d N      : set ROOT gDebug flag to N" << endl ;
      cout << " " << endl ;
      return 0 ;
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

  // Disable caching of complex error function calculation, as we don't 
  // want to write out the cache file as part of the validation procedure
  RooMath::cacheCERF(kFALSE) ;

  gBenchmark = new TBenchmark();
  stressRooFit(refFileName.c_str(),doWrite,doVerbose,oneTest,dryRun,doDump,doVectorStore);  
  return 0;
}

#endif
