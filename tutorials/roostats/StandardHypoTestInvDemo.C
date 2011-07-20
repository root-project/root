// Standard tutorial macro for performing an inverted  hypothesis test 
//
// This macro will perform a scan of tehe p-values for computing the limit
// 

#include "TFile.h"
#include "RooWorkspace.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooStats/ModelConfig.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TCanvas.h"
#include "TLine.h"

#include "RooStats/HybridCalculator.h"
#include "RooStats/FrequentistCalculator.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/HypoTestPlot.h"

#include "RooStats/NumEventsTestStat.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/SimpleLikelihoodRatioTestStat.h"
#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"
#include "RooStats/MaxLikelihoodEstimateTestStat.h"

#include "RooStats/HypoTestInverter.h"
#include "RooStats/HypoTestInverterResult.h"
#include "RooStats/HypoTestInverterPlot.h"

using namespace RooFit;
using namespace RooStats;


bool plotHypoTestResult = true; 
bool useProof = true;
bool optimize = false;
bool writeResult = false;
int nworkers = 4;
bool rebuild = false;
int nToyToRebuild = 100;

// internal routine to run the inverter
HypoTestInverterResult * RunInverter(RooWorkspace * w, const char * modelSBName, const char * modelBName, const char * dataName,
                                     int type,  int testStatType, int npoints, double poimin, double poimax, int ntoys, 
                                     bool useCls, bool useNumberCounting, const char * nuisPriorName);




void StandardHypoTestInvDemo(const char * fileName =0,
                             const char * wsName = "combined",
                             const char * modelSBName = "ModelConfig",
                             const char * modelBName = "",
                             const char * dataName = "obsData",                 
                             int calculatorType = 0,
                             int testStatType = 3, 
                             bool useCls = true ,  
                             int npoints = 5,   
                             double poimin = 0,  
                             double poimax = 5, 
                             int ntoys=1000,
                             bool useNumberCounting = false,
                             const char * nuisPriorName = 0)    
{
/*

   Other Parameter to pass in tutorial
   apart from standard for filename, ws, modelconfig and data

    type = 0 Freq calculator 
    type = 1 Hybrid 

    testStatType = 0 LEP
                 = 1 Tevatron 
                 = 2 Profile Likelihood
                 = 3 Profile Likelihood one sided (i.e. = 0 if mu < mu_hat)

    useCLs          scan for CLs (otherwise for CLs+b)    

    npoints:        number of points to scan , for autoscan set npoints = -1 

    poimin,poimax:  min/max value to scan in case of fixed scans 
                    (if min >= max, try to find automatically)                           

    ntoys:         number of toys to use 

    useNumberCounting:  set to true when using number counting events 

    nuisPriorName:   name of prior for the nnuisance. This is often expressed as constraint term in the global model
                     It is needed only when using the HybridCalculator (type=1)
                     If not given by default the prior pdf from ModelConfig is used. 

    extra options are available as global paramters of the macro. They are: 

    plotHypoTestResult   plot result of tests at each point (TS distributions) 
    useProof = true;
    writeResult = true;
    nworkers = 4;


   */

   if (fileName==0) { 
      fileName = "results/example_combined_GaussExample_model.root";
      std::cout << "Use standard file generated with HistFactory :" << fileName << std::endl;
   }
   TFile * file = new TFile(fileName); 

   RooWorkspace * w = dynamic_cast<RooWorkspace*>( file->Get(wsName) );
   HypoTestInverterResult * r = 0; 
   std::cout << w << "\t" << fileName << std::endl;
   if (w != NULL) {
      r = RunInverter(w, modelSBName, modelBName, dataName, calculatorType, testStatType, npoints, poimin, poimax,  
                      ntoys, useCls, useNumberCounting, nuisPriorName );    
      if (!r) { 
         std::cerr << "Error running the HypoTestInverter - Exit " << std::endl;
         return;          
      }
   }
   else 
   { 
      // case workspace is not present look for the inverter result
      std::cout << "Reading an HypoTestInverterResult with name " << wsName << " from file " << fileName << std::endl;
      r = dynamic_cast<HypoTestInverterResult*>( file->Get(wsName) ); //
      if (!r) { 
         std::cerr << "File " << fileName << " does not contain a workspace or an HypoTestInverterResult - Exit " 
                   << std::endl;
         file->ls();
         return; 
      }
   }		
      		

   double upperLimit = r->UpperLimit();
   double ulError = r->UpperLimitEstimatedError();


   std::cout << "The computed upper limit is: " << upperLimit << " +/- " << ulError << std::endl;
 
   const int nEntries = r->ArraySize();


   const char *  typeName = (calculatorType == 0) ? "Frequentist" : "Hybrid";
   const char * resultName = (w) ? w->GetName() : r->GetName();
   TString plotTitle = TString::Format("%s CL Scan for workspace %s",typeName,resultName);
   HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot",plotTitle,r);
   plot->Draw("CLb 2CL");  // plot all and Clb

   if (plotHypoTestResult) { 
      TCanvas * c2 = new TCanvas();
      c2->Divide( 2, TMath::Ceil(nEntries/2));
      for (int i=0; i<nEntries; i++) {
         c2->cd(i+1);
         SamplingDistPlot * pl = plot->MakeTestStatPlot(i);
         pl->SetLogYaxis(true);
         pl->Draw();
      }
   }


   std::cout << " expected limit (median) " <<  r->GetExpectedUpperLimit(0) << std::endl;
   std::cout << " expected limit (-1 sig) " << r->GetExpectedUpperLimit(-1) << std::endl;
   std::cout << " expected limit (+1 sig) " << r->GetExpectedUpperLimit(1) << std::endl;


   if (w != NULL && writeResult) {

      // write to a file the results
      const char *  calcType = (calculatorType == 0) ? "Freq" : "Hybr";
      const char *  limitType = (useCls) ? "CLs" : "Cls+b";
      const char * scanType = (npoints < 0) ? "auto" : "grid";
      TString resultFileName = TString::Format("%s_%s_%s_ts%d_",calcType,limitType,scanType,testStatType);      
      resultFileName += fileName;
      
      TFile * fileOut = new TFile(resultFileName,"RECREATE");
      r->Write();
      fileOut->Close();                                                                     
   }   

}


// internal routine to run the inverter
HypoTestInverterResult *  RunInverter(RooWorkspace * w, const char * modelSBName, const char * modelBName, 
                                      const char * dataName, int type,  int testStatType, 
                                      int npoints, double poimin, double poimax, 
                                      int ntoys, bool useCls, bool useNumberCounting, const char * nuisPriorName ) 
{

   std::cout << "Running HypoTestInverter on the workspace " << w->GetName() << std::endl;

   w->Print();


   RooAbsData * data = w->data(dataName); 
   if (!data) { 
      Error("StandardHypoTestDemo","Not existing data %s",dataName);
      return 0;
   }
   else 
      std::cout << "Using data set " << dataName << std::endl;

   
   // get models from WS
   // get the modelConfig out of the file
   ModelConfig* bModel = (ModelConfig*) w->obj(modelBName);
   ModelConfig* sbModel = (ModelConfig*) w->obj(modelSBName);

   if (!sbModel) {
      Error("StandardHypoTestDemo","Not existing ModelConfig %s",modelSBName);
      return 0;
   }
   // check the model 
   if (!sbModel->GetPdf()) { 
      Error("StandardHypoTestDemo","Model %s has no pdf ",modelSBName);
      return 0;
   }
   if (!sbModel->GetParametersOfInterest()) {
      Error("StandardHypoTestDemo","Model %s has no poi ",modelSBName);
      return 0;
   }
   if (!sbModel->GetParametersOfInterest()) {
      Error("StandardHypoTestInvDemo","Model %s has no poi ",modelSBName);
      return 0;
   }
   if (!sbModel->GetSnapshot() ) { 
      Info("StandardHypoTestInvDemo","Model %s has no snapshot  - make one using model poi",modelSBName);
      sbModel->SetSnapshot( *sbModel->GetParametersOfInterest() );
   }


   if (!bModel || bModel == sbModel) {
      Info("StandardHypoTestInvDemo","The background model %s does not exist",modelBName);
      Info("StandardHypoTestInvDemo","Copy it from ModelConfig %s and set POI to zero",modelSBName);
      bModel = (ModelConfig*) sbModel->Clone();
      bModel->SetName(TString(modelSBName)+TString("_with_poi_0"));      
      RooRealVar * var = dynamic_cast<RooRealVar*>(bModel->GetParametersOfInterest()->first());
      if (!var) return 0;
      double oldval = var->getVal();
      var->setVal(0);
      bModel->SetSnapshot( RooArgSet(*var)  );
      var->setVal(oldval);
   }
   else { 
      if (!bModel->GetSnapshot() ) { 
         Info("StandardHypoTestInvDemo","Model %s has no snapshot  - make one using model poi and 0 values ",modelBName);
         RooRealVar * var = dynamic_cast<RooRealVar*>(bModel->GetParametersOfInterest()->first());
         if (var) { 
            double oldval = var->getVal();
            var->setVal(0);
            bModel->SetSnapshot( RooArgSet(*var)  );
            var->setVal(oldval);
         }
         else { 
            Error("StandardHypoTestInvDemo","Model %s has no valid poi",modelBName);
            return 0;
         }         
      }
   }


   SimpleLikelihoodRatioTestStat slrts(*sbModel->GetPdf(),*bModel->GetPdf());
   if (sbModel->GetSnapshot()) slrts.SetNullParameters(*sbModel->GetSnapshot());
   if (bModel->GetSnapshot()) slrts.SetAltParameters(*bModel->GetSnapshot());

   // ratio of profile likelihood - need to pass snapshot for the alt
   RatioOfProfiledLikelihoodsTestStat 
      ropl(*sbModel->GetPdf(), *bModel->GetPdf(), bModel->GetSnapshot());
   ropl.SetSubtractMLE(false);
   
   ProfileLikelihoodTestStat profll(*sbModel->GetPdf());
   if (testStatType == 3) profll.SetOneSided(1);
   if (optimize) profll.SetReuseNLL(true);

   TestStatistic * testStat = &slrts;
   if (testStatType == 1) testStat = &ropl;
   if (testStatType == 2 || testStatType == 3) testStat = &profll;
  
   
   HypoTestCalculatorGeneric *  hc = 0;
   if (type == 0) hc = new FrequentistCalculator(*data, *bModel, *sbModel);
   else hc = new HybridCalculator(*data, *bModel, *sbModel);

   ToyMCSampler *toymcs = (ToyMCSampler*)hc->GetTestStatSampler();
   if (useNumberCounting) toymcs->SetNEventsPerToy(1);
   toymcs->SetTestStatistic(testStat);
   if (optimize) toymcs->SetUseMultiGen(true);


   if (type == 1) { 
      HybridCalculator *hhc = (HybridCalculator*) hc;
      hhc->SetToys(ntoys,ntoys); 

      // check for nuisance prior pdf in case of nuisance parameters 
      if (bModel->GetNuisanceParameters() || sbModel->GetNuisanceParameters() ) {
         RooAbsPdf * nuisPdf = 0; 
         if (nuisPriorName) nuisPdf = w->pdf(nuisPriorName);
         // use prior defined first in bModel (then in SbModel)
         if (!nuisPdf)  { 
            Info("StandardHypoTestInvDemo","No nuisance pdf given for the HybridCalculator - try to use the prior pdf from the model");
            nuisPdf = (bModel->GetPriorPdf() ) ?  bModel->GetPriorPdf() : sbModel->GetPriorPdf();
         }
         if (!nuisPdf) { 
            Error("StandardHypoTestInvDemo","Cannnot run Hybrid calculator because no prior on the nuisance parameter is specified");
            return 0;
         }
         const RooArgSet * nuisParams = (bModel->GetNuisanceParameters() ) ? bModel->GetNuisanceParameters() : sbModel->GetNuisanceParameters();
         RooArgSet * np = nuisPdf->getObservables(*nuisParams);
         if (np->getSize() == 0) { 
            Warning("StandardHypoTestInvDemo","Prior nuisance does not depend on nuisance parameters. They will be smeared in their full range");
         }
         delete np;
         hhc->ForcePriorNuisanceAlt(*nuisPdf);
         hhc->ForcePriorNuisanceNull(*nuisPdf);
      }
   } 
   else 
      ((FrequentistCalculator*) hc)->SetToys(ntoys,ntoys); 

   // Get the result
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);


   TStopwatch tw; tw.Start(); 
   const RooArgSet * poiSet = sbModel->GetParametersOfInterest();
   RooRealVar *poi = (RooRealVar*)poiSet->first();

   // fit the data first
   sbModel->GetPdf()->fitTo(*data);
   double poihat  = poi->getVal();


   HypoTestInverter calc(*hc);
   calc.SetConfidenceLevel(0.95);

   calc.UseCLs(useCls);
   calc.SetVerbose(true);

   // can speed up using proof-lite
   if (useProof && nworkers > 1) { 
      ProofConfig pc(*w, nworkers, "", kFALSE);
      toymcs->SetProofConfig(&pc);    // enable proof
   }

   
   if (npoints > 0) {
      if (poimin >= poimax) { 
         // if no min/max given scan between MLE and +4 sigma 
         poimin = int(poihat);
         poimax = int(poihat +  4 * poi->getError());
      }
      std::cout << "Doing a fixed scan  in interval : " << poimin << " , " << poimax << std::endl;
      calc.SetFixedScan(npoints,poimin,poimax);
   }
   else { 
      //poi->setMax(10*int( (poihat+ 10 *poi->getError() )/10 ) );
      std::cout << "Doing an  automatic scan  in interval : " << poi->getMin() << " , " << poi->getMax() << std::endl;
   }

   HypoTestInverterResult * r = calc.GetInterval();

   if (rebuild) {
      calc.SetCloseProof(1);
      SamplingDistribution * limDist = calc.GetUpperLimitDistribution(true,nToyToRebuild);
      if (limDist) { 
         std::cout << "expected up limit " << limDist->InverseCDF(0.5) << " +/- " 
                   << limDist->InverseCDF(0.16) << "  " 
                   << limDist->InverseCDF(0.84) << "\n"; 
      }
      else 
         std::cout << "ERROR : failed to re-build distributions " << std::endl; 
   }

   //update r to a new re-freshed copied
   r = calc.GetInterval();

   return r; 
}

void ReadResult(const char * fileName, const char * resultName="", bool useCLs=true) { 
   // read a previous stored result from a file given the result name

   StandardHypoTestInvDemo(fileName, resultName,"","","",0,0,useCLs);
}

int main() {
   StandardHypoTestInvDemo();
}
