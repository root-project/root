// C/C++ headers
#include <iostream>
#include <stdio.h>

// ROOT headers
#include "TFile.h"
#include "TString.h"
#include "TSystem.h"
#include "TMath.h"
#include "TH1F.h"
#include "TMinuit.h"

// RooFit headers
#include "RooUnitTest.h"
#include "RooWorkspace.h"
#include "RooArgSet.h"
#include "RooLinkedListIter.h"
#include "RooAbsPdf.h"
#include "RooDataSet.h"

// RooStats header(s)
#include "RooStats/ModelConfig.h"
#include "RooStats/RooStatsUtils.h"

#include "stressHistFactory_models.cxx"

using namespace std;
using namespace RooFit;
using namespace RooStats;

class PdfComparison : public RooUnitTest {
private:
  TString fTestDirectory;
  TString fOldDirectory;  // old directory where test is started
  Double_t fTolerance;
public:
  PdfComparison(
    TFile* refFile,
    Bool_t writeRef,
    Int_t verbose
    ) :
    RooUnitTest("PDF comparison for HistFactory", refFile, writeRef, verbose),
    fTestDirectory("HistFactoryTest"),
    fTolerance(1e-3)
  {
     fOldDirectory = gSystem->pwd();
  }

  ~PdfComparison()
  {
     // delete temmporary directory if in not verbose mode
     if (_verb == 0) {
        TString cmd = "rm -rf ";
        cmd += fTestDirectory;
        int ret = gSystem->Exec(cmd);
        if (ret != 0) Error("PdfComparison","Error removing test directory %s",fTestDirectory.Data());
     }
  }

  Bool_t isTestAvailable()
  {
     bool ret = true;
     ret &= CreateTestDirectory();
     if (!ret) { Error("PdfComparison","Error creating test directory"); return ret; }
     ret &= CopyTestFiles();
     if (!ret) { Error("PdfComparison","Error copying test files from $ROOTSYS/test/HistFactoryTest.tar"); return ret; }
     ret &= UnpackTestFiles();
     if (!ret) Error("PdfComparison","Error unpacking test file HistFactoryTest.tar");
     return ret;
  }

  Bool_t testCode()
  {
    // print information where the temporary test/log files are placed
    if(_verb > 0)
      std::cout << "using test directory: " << fTestDirectory << std::endl;

    // dump histfactory output into a file
    gSystem->RedirectOutput(fTestDirectory + "/API_vs_XML_test.log","a");

    // build model using the API
    // create and move to a temporary directory to run hist2workspace
    gSystem->ChangeDirectory(fTestDirectory + "/API/");
    buildAPI_XML_TestModel("API_XML_TestModel");

    // get API workspace and ModelConfig
    TFile* pAPIFile = TFile::Open("API_XML_TestModel_combined_Test_model.root");
    if(!pAPIFile || pAPIFile->IsZombie()) {
       Error("testCode","Error opening the file API_XML_TestModel_combined_Test_model.root");
       return kFALSE;
    }

    RooWorkspace* pWS_API = (RooWorkspace*)pAPIFile->Get("combined");
    if(!pWS_API) {
       Error("testCode","Error retrieving the workspace combined");
       return kFALSE;
    }

    ModelConfig* pMC_API = (ModelConfig*)pWS_API->obj("ModelConfig");
    if(!pMC_API) {
       Error("testCode","Error retrieving the ModelConfig");
       return kFALSE;
    }
    // build model using XML files
    gSystem->ChangeDirectory(fTestDirectory + "/XML/");
    // be sure libraries are found for running hist2workspace
    gSystem->AddDynamicPath("$ROOTSYS/lib");
    TString cmd = "$ROOTSYS/bin/hist2workspace config/Measurement.xml";
    int ret = gSystem->Exec(cmd);
    if (ret != 0) {
       Error("testCode","Error running hist2workspace");
       return kFALSE;
    }

    // get XML workspace and ModelConfig
    TFile* pXMLFile = TFile::Open("results/API_XML_TestModel_combined_Test_model.root");
    if(!pXMLFile || pXMLFile->IsZombie()) {
       Error("testCode","Error opening the file results/API_XML_TestModel_combined_Test_model.root");
       return kFALSE;
    }

    RooWorkspace* pWS_XML = (RooWorkspace*)pXMLFile->Get("combined");
    if(!pWS_XML) {
       Error("testCode","Error retrieving the workspace combined");
       return kFALSE;
    }

    ModelConfig* pMC_XML = (ModelConfig*)pWS_XML->obj("ModelConfig");
    if(!pMC_XML) {
       Error("testCode","Error retrieving the ModelConfig");
       return kFALSE;
    }

    // cancel redirection
    gSystem->RedirectOutput(0);

    // change working directory to original one
    gSystem->ChangeDirectory(fOldDirectory);

    // compare data
    if(pWS_API->data("obsData"))
    {
      assert(pWS_XML->data("obsData"));
      if(!CompareData(*pWS_API->data("obsData"),*pWS_XML->data("obsData")))
         return kFALSE;
    }
    else
      return kFALSE;

    if(pWS_API->data("asimovData"))
    {
      assert(pWS_XML->data("asimovData"));
      if(!CompareData(*pWS_API->data("asimovData"),*pWS_XML->data("asimovData")))
        return kFALSE;
    }
    else
      return kFALSE;

    // compare sets of parameters
    if(pMC_API->GetParametersOfInterest())
    {
      assert(pMC_XML->GetParametersOfInterest());
      if(_verb > 0)
         Info("testCode","comparing PoIs");
      if(!CompareParameters(*pMC_API->GetParametersOfInterest(),*pMC_XML->GetParametersOfInterest()))
         return kFALSE;
    }
     else
      assert(!pMC_XML->GetParametersOfInterest());

    if(pMC_API->GetObservables())
    {
      assert(pMC_XML->GetObservables());
      if(_verb > 0)
         Info("testCode","comparing observables");
      if(!CompareParameters(*pMC_API->GetObservables(),*pMC_XML->GetObservables()))
         return kFALSE;
    }
    else
      assert(!pMC_XML->GetObservables());

    if(pMC_API->GetGlobalObservables())
    {
      assert(pMC_XML->GetGlobalObservables());
      if(_verb > 0)
         Info("testCode","comparing global observables");
      if(!CompareParameters(*pMC_API->GetGlobalObservables(),*pMC_XML->GetGlobalObservables()))
         return kFALSE;
    }
    else
      assert(!pMC_XML->GetGlobalObservables());

    if(pMC_API->GetConditionalObservables())
    {
      assert(pMC_XML->GetConditionalObservables());
      if(_verb > 0)
         Info("testCode","comparing conditional observables");
      if(!CompareParameters(*pMC_API->GetConditionalObservables(),*pMC_XML->GetConditionalObservables()))
         return kFALSE;
    }
    else
      assert(!pMC_XML->GetConditionalObservables());

    if(pMC_API->GetNuisanceParameters())
    {
      assert(pMC_XML->GetNuisanceParameters());
      if(_verb > 0)
         Info("testCode","comparing nuisance parameters");
      if(!CompareParameters(*pMC_API->GetNuisanceParameters(),*pMC_XML->GetNuisanceParameters()))
         return kFALSE;
    }
    else
      assert(!pMC_XML->GetNuisanceParameters());

    // compare pdfs
    assert(pMC_API->GetPdf() && pMC_XML->GetPdf());
    RooArgSet* pObservables = (RooArgSet*)(pMC_API->GetObservables()->snapshot());
    RooArgSet* pGlobalObservables = (RooArgSet*)(pMC_API->GetGlobalObservables()->snapshot());
    if(pGlobalObservables)
    {
      pObservables->addOwned(*pGlobalObservables);
    }

    if(_verb > 0)
      Info("testCode","comparing PDFs");
    if(!ComparePDF(*pMC_API->GetPdf(),*pMC_XML->GetPdf(),*pObservables,*pWS_API->data("obsData")))
    {
      delete pObservables;
      // delete pGlobalObservables;
      return kFALSE;
    }

    // clean up
    delete pObservables;
    // delete pGlobalObservables;

    return kTRUE;
  }

private:
  Bool_t CreateTestDirectory()
  {
    // use trick to get unique, unoccupied file name as test directory name
    FILE* pTmpFile = gSystem->TempFileName(fTestDirectory,gSystem->TempDirectory());
    fclose(pTmpFile);
    gSystem->Unlink(fTestDirectory.Data());

    // try to create test directory with obtained temp name
    return (gSystem->MakeDirectory(fTestDirectory.Data()) == 0);
  }

  Bool_t CopyTestFiles()
  {
    return (gSystem->CopyFile(gSystem->ExpandPathName("$ROOTSYS/test/HistFactoryTest.tar"),(fTestDirectory + "/HistFactoryTest.tar").Data(),kTRUE) == 0);
  }

  Bool_t UnpackTestFiles()
  {
    TString cmd = "tar -xf ";
    TString tarFile = fTestDirectory + "/HistFactoryTest.tar";
    cmd.Append(tarFile);
    gSystem->ChangeDirectory(gSystem->DirName(tarFile));

    return (gSystem->Exec(cmd) == 0);
  }

  Bool_t CompareData(const RooAbsData& rData1,const RooAbsData& rData2)
  {
    if(rData1.numEntries() != rData2.numEntries())
    {
       Warning("CompareData","data sets have different numbers of entries: %d vs %d",rData1.numEntries(),rData2.numEntries());
       return kFALSE;
    }

    if(rData1.sumEntries() != rData2.sumEntries())
    {
      Warning("CompareData","data sets have different sums of weights");
      return kFALSE;
    }

    const RooArgSet* set1 = rData1.get();
    const RooArgSet* set2 = rData2.get();

    if(!CompareParameters(*set1,*set2))
      return kFALSE;

    RooLinkedListIter it = set1->iterator();
    RooAbsArg* arg = 0;
    while((arg = (RooAbsArg*)it.Next()))
    {
      RooRealVar * par = dynamic_cast<RooRealVar*>(arg);
      if (!par) continue;  // do not test RooCategory
      if(!TMath::AreEqualAbs(rData1.mean(*par),rData2.mean(*par),fTolerance))
      {
         Warning("CompareData","data sets have different means for \"%s\": %.3f vs %.3f",par->GetName(),rData1.mean(*par),rData2.mean(*par));
         return kFALSE;
      }

      if(!TMath::AreEqualAbs(rData1.sigma(*par),rData2.sigma(*par),fTolerance))
      {
         Warning("CompareData","data sets have different sigmas for \"%s\": %.3f vs %.3f",par->GetName(),rData1.sigma(*par),rData2.sigma(*par));
         return kFALSE;
      }
    }

    return kTRUE;
  }

  Bool_t CompareParameters(const RooArgSet& rPars1, const RooArgSet& rPars2,Bool_t bAllowForError = kFALSE)
  {
    if(rPars1.getSize() != rPars2.getSize())
    {
      Warning("CompareParameters","got different numbers of parameters: %d vs %d",rPars1.getSize(),rPars2.getSize());
      return kFALSE;
    }

    RooLinkedListIter it = rPars1.iterator();
    RooRealVar* arg1 = 0;
    RooRealVar* arg2 = 0;
    TObject* obj = 0;
    while((obj = it.Next()))
    {
      // checks only for RooRealVars implemented
      arg1 = dynamic_cast<RooRealVar*>(obj);
      if(!arg1)
         continue;

      arg2 = (RooRealVar*)rPars2.find(arg1->GetName());

      if(!arg2)
      {
         Warning("CompareParameters","did not find observable with name \"%s\"",arg1->GetName());
         return kFALSE;
      }

      if(!TMath::AreEqualAbs(arg1->getMin(),arg2->getMin(),fTolerance))
      {
         Warning("CompareParameters","parameters with name \"%s\" have different minima: %.3f vs %.3f",arg1->GetName(),arg1->getMin(),arg2->getMin());
         return kFALSE;
      }

      if(!TMath::AreEqualAbs(arg1->getMax(),arg2->getMax(),fTolerance))
      {
         Warning("CompareParameters","parameters with name \"%s\" have different maxima: %.3f vs %.3f",arg1->GetName(),arg1->getMax(),arg2->getMax());
         return kFALSE;
      }

      if(arg1->getBins() != arg2->getBins())
      {
         Warning("CompareParameters","parameters with name \"%s\" have different number of bins: %d vs %d",arg1->GetName(),arg1->getBins(),arg2->getBins());
         return kFALSE;
      }

      if(arg1->isConstant() != arg2->isConstant())
      {
         Warning("CompareParameters","parameters with name \"%s\" have different constness",arg1->GetName());
         return kFALSE;
      }

      if(bAllowForError)
      {
         if(!TMath::AreEqualAbs(arg1->getVal(),arg2->getVal(), TMath::Max(fTolerance,0.1*TMath::Min(arg1->getError(),arg2->getError()))))
         {
            Warning("CompareParameters","parameters with name \"%s\" have different values: %.3f +/- %.3f vs %.3f +/- %.3f",arg1->GetName(),arg1->getVal(),arg1->getError(),arg2->getVal(),arg2->getError());
            return kFALSE;
         }
      }
      else
       {
          if(!TMath::AreEqualAbs(arg1->getVal(),arg2->getVal(),fTolerance))
          {
             Warning("CompareParameters","parameters with name \"%s\" have different values: %.3f vs %.3f",arg1->GetName(),arg1->getVal(),arg2->getVal());
             return kFALSE;
          }

          if(!TMath::AreEqualAbs(arg1->getError(),arg2->getError(),fTolerance))
          {
             Warning("CompareParameters","parameters with name \"%s\" have different errors: %.3f vs %.3f",arg1->GetName(),arg1->getError(),arg2->getError());
             return kFALSE;
          }
       }
    }

     return kTRUE;
  }

  Bool_t ComparePDF(RooAbsPdf& rPDF1,RooAbsPdf& rPDF2,const RooArgSet& rAllObservables,RooAbsData& rTestData)
  {
    // options
    const Int_t iSamplingPoints = 100;

    // get variables
    RooArgSet* pVars1 = rPDF1.getVariables();
    RooArgSet* pVars2 = rPDF2.getVariables();

    if(!CompareParameters(*pVars1,*pVars2))
    {
      Warning("ComparePDF","variable sets for PDFs failed check");
      return kFALSE;
    }

    RooDataSet* pSamplingPoints = rPDF1.generate(rAllObservables,NumEvents(iSamplingPoints));
    TH1F* h_diff = new TH1F("h_diff","relative difference between both PDF;#Delta;Points / 1e-4",200,-0.01,0.01);

    float fPDF1value, fPDF2value;
    for(Int_t i = 0; i < pSamplingPoints->numEntries(); ++i)
    {
      *pVars1 = *pSamplingPoints->get(i);
      *pVars2 = *pSamplingPoints->get(i);

      fPDF1value = rPDF1.getVal();
      fPDF2value = rPDF2.getVal();

      float diff = (fPDF1value - fPDF2value);
      if (fPDF1value!=0.f) diff /= fPDF1value; // Protect against NaN
      h_diff->Fill(diff);
    }

    Bool_t bResult = kTRUE;

    // no deviations > 1%
    if((h_diff->GetBinContent(0) > 0) || (h_diff->GetBinContent(h_diff->GetNbinsX()) > 0))
    {
      Warning("ComparePDF","PDFs deviate more than 1%% for individual parameter point(s)");
      bResult = kFALSE;
    }

    // mean deviation < 0.1%
    if(h_diff->GetMean() > 1e-3)
    {
      Warning("ComparePDF","PDFs deviate on average more than 0.1%%");
      bResult = kFALSE;
    }

    // clean up
    delete pSamplingPoints;
    delete h_diff;

    if(!bResult)
      return kFALSE;

    // check fit result to test data
    *pVars1 = *pVars2;

    // do the fit
    std::string minimizerType = "Minuit2";
    int prec = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kFatal;
    if (gSystem->Load("libMinuit2") < 0) minimizerType = "Minuit";
    gErrorIgnoreLevel=prec;

    RooFitResult* r1 = rPDF1.fitTo(rTestData,Save(), RooFit::Minimizer(minimizerType.c_str()));
    //L.M:  for minuit we need ot rest otherwise fit could fail
    if (minimizerType == "Minuit") {
       if (gMinuit) { delete gMinuit; gMinuit=0; }
    }
    RooFitResult* r2 = rPDF2.fitTo(rTestData,Save(), RooFit::Minimizer(minimizerType.c_str()));

    if(_verb > 0)
    {
      r1->Print("v");
      r2->Print("v");
    }

    if(!TMath::AreEqualAbs(r1->minNll(),r2->minNll(),0.05))
    {
      Warning("ComparePDF","likelihood end up in different minima: %.3f vs %.3f",r1->minNll(),r2->minNll());
      return kFALSE;
    }

    if(!CompareParameters(*pVars1,*pVars2,kTRUE))
    {
      Warning("ComparePDF","variable sets of PDFs differ after fit to test data");
      return kFALSE;
    }

    return kTRUE;
  }
};
