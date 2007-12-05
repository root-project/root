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

      if (!iter2->first->isIdentical(*bmark,1e-6,1e-4)) {
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


#include "TestBasic1.cxx"
#include "TestBasic2.cxx"
#include "TestBasic3.cxx"
#include "TestBasic4.cxx"
#include "TestBasic5.cxx"
#include "TestBasic6.cxx"
#include "TestBasic7.cxx"
#include "TestBasic8.cxx"
#include "TestBasic9.cxx"
#include "TestBasic10.cxx"
#include "TestBasic11.cxx"
#include "TestBasic12.cxx"
#include "TestBasic13.cxx"
#include "TestBasic14.cxx"
#include "TestBasic15.cxx"
#include "TestBasic16.cxx"
#include "TestBasic17.cxx"
#include "TestBasic18.cxx"
#include "TestBasic19.cxx"
#include "TestBasic20.cxx"
#include "TestBasic21.cxx"
#include "TestBasic22.cxx"


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
    StatusPrint(11,"Workspace persistence",test11.runTest()) ; // to do
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
    printf("*  %s\n",line);
    gSystem->ClosePipe(fp);
  } else {
    const Char_t *os = gSystem->Getenv("OS");
    if (!os) printf("*  Windows 95\n");
    else     printf("*  %s %s \n",os,gSystem->Getenv("PROCESSOR_IDENTIFIER"));
  }
  
  printf("******************************************************************\n");
  gBenchmark->Print("StressFit");
#ifdef __CINT__
  Double_t reftime = 86.34; //macbrun interpreted
#else
  Double_t reftime = 12.07; //macbrun compiled
#endif
  const Double_t rootmarks = 800*reftime/gBenchmark->GetCpuTime("StressRooFit");
  
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

  }

  gBenchmark = new TBenchmark();
  stressRooFit(refFileName.c_str(),doWrite,doVerbose,oneTest,dryRun);  
  return 0;
}

#endif
