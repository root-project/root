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
#include "TStopwatch.h"
#include "TROOT.h"

#include "RooStats/HybridCalculator.h"
#include "RooStats/FrequentistCalculator.h"
#include "RooStats/AsymptoticCalculator.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/HypoTestPlot.h"
#include "RooStats/RooStatsUtils.h"

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


bool plotHypoTestResult = true;          // plot test statistic result at each point
bool writeResult = false;                // write HypoTestInverterResult in a file 
bool optimize = true;                    // optmize evaluation of test statistic 
bool useVectorStore = true;              // convert data to use new roofit data store 
bool generateBinned = false;             // generate binned data sets 
double nToysRatio = 2;                   // ratio Ntoys S+b/ntoysB
double maxPOI = -1;                      // max value used of POI (in case of auto scan) 
bool useProof = true;                    // use Proof Light when using toys (for freq or hybrid)
int nworkers = 4;                        // number of worker for Proof
bool rebuild = false;                    // re-do extra toys for computing expected limits and rebuild test stat
                                         // distributions (N.B this requires much more CPU (factor is equivalent to nToyToRebuild)
int nToyToRebuild = 100;                 // number of toys used to rebuild 

const char * massValue = 0;              // extra string to tag output file of result 
TString  minimizerType;                  // minimizer type (default is what is in ROOT::Math::MinimizerOptions::DefaultMinimizerType()
int   printLevel = 0;                    // print level for debugging PL test statistics and calculators   




// internal routine to run the inverter
HypoTestInverterResult * RunInverter(RooWorkspace * w, const char * modelSBName, const char * modelBName, const char * dataName,
                                     int type,  int testStatType, int npoints, double poimin, double poimax, int ntoys, 
                                     bool useCls, bool useNumberCounting, const char * nuisPriorName);




void StandardHypoTestInvDemo(const char * infile =0,
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
    type = 1 Hybrid calculator
    type = 2 Asymptotic calculator  

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

    extra options are available as global paramwters of the macro. They major ones are: 

    plotHypoTestResult   plot result of tests at each point (TS distributions) (defauly is true)
    useProof             use Proof   (default is true) 
    writeResult          write result of scan (default is true)
    rebuild              rebuild scan for expected limits (require extra toys) (default is false)
    generateBinned       generate binned data sets for toys (default is false) - be careful not to activate with 
                         a too large (>=3) number of observables 
    nToyRatio            ratio of S+B/B toys (default is 2)
    

   */

   
   TString fileName(infile);
   if (fileName.IsNull()) { 
      fileName = "results/example_combined_GaussExample_model.root";
      std::cout << "Use standard file generated with HistFactory : " << fileName << std::endl;
   }

   // open file and check if input file exists
   TFile * file = TFile::Open(fileName); 

  // if input file was specified but not found, quit
  if(!file && !TString(infile).IsNull()){
     cout <<"file " << fileName << " not found" << endl;
     return;
  } 

  // if default file not found, try to create it
  if(!file ){
    // Normally this would be run on the command line
    cout <<"will run standard hist2workspace example"<<endl;
    gROOT->ProcessLine(".! prepareHistFactory .");
    gROOT->ProcessLine(".! hist2workspace config/example.xml");
    cout <<"\n\n---------------------"<<endl;
    cout <<"Done creating example input"<<endl;
    cout <<"---------------------\n\n"<<endl;

    // now try to access the file again
    file = TFile::Open(fileName);

  }

  if(!file){
    // if it is still not there, then we can't continue
    cout << "Not able to run hist2workspace to create example input" <<endl;
    return;
  }



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

   // compute expected limit
   std::cout << " expected limit (median) " <<  r->GetExpectedUpperLimit(0) << std::endl;
   std::cout << " expected limit (-1 sig) " << r->GetExpectedUpperLimit(-1) << std::endl;
   std::cout << " expected limit (+1 sig) " << r->GetExpectedUpperLimit(1) << std::endl;
   std::cout << " expected limit (-2 sig) " << r->GetExpectedUpperLimit(-2) << std::endl;
   std::cout << " expected limit (+2 sig) " << r->GetExpectedUpperLimit(2) << std::endl;


   // write result in a file 
   if (r != NULL && writeResult) {

      // write to a file the results
      const char *  calcType = (calculatorType == 0) ? "Freq" : (calculatorType == 1) ? "Hybr" : "Asym";
      const char *  limitType = (useCls) ? "CLs" : "Cls+b";
      const char * scanType = (npoints < 0) ? "auto" : "grid";
      TString resultFileName = TString::Format("%s_%s_%s_ts%d_",calcType,limitType,scanType,testStatType);      
      //strip the / from the filename
      if (massValue) {
         resultFileName += massValue;
         resultFileName += "_";
      }

      TString name = fileName; 
      name.Replace(0, name.Last('/')+1, "");
      resultFileName += name;
         
      TFile * fileOut = new TFile(resultFileName,"RECREATE");
      r->Write();
      fileOut->Close();                                                                     
   }   

 
   // plot the result ( p values vs scan points) 
   std::string typeName = "Frequentist";
   if (calculatorType == 1 )
      typeName = "Hybrid";   
   else if (calculatorType == 2 ) { 
      typeName = "Asymptotic";
      plotHypoTestResult = false; 
   }
      
   const char * resultName = (w) ? w->GetName() : r->GetName();
   TString plotTitle = TString::Format("%s CL Scan for workspace %s",typeName.c_str(),resultName);
   HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot",plotTitle,r);
   plot->Draw("CLb 2CL");  // plot all and Clb
   
   const int nEntries = r->ArraySize();

   // plot test statistics distributions for the two hypothesis 
   if (plotHypoTestResult) { 
      TCanvas * c2 = new TCanvas();
      if (nEntries > 1) { 
         int ny = TMath::CeilNint( sqrt(nEntries) );
         int nx = TMath::CeilNint(double(nEntries)/ny);
         c2->Divide( nx,ny);
      }
      for (int i=0; i<nEntries; i++) {
         if (nEntries > 1) c2->cd(i+1);
         SamplingDistPlot * pl = plot->MakeTestStatPlot(i);
         pl->SetLogYaxis(true);
         pl->Draw();
      }
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

   if (useVectorStore) { 
      RooAbsData::defaultStorageType = RooAbsData::Vector;
      data->convertToVectorStore() ;
   }

   
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
   if (!sbModel->GetObservables()) {
      Error("StandardHypoTestInvDemo","Model %s has no observables ",modelSBName);
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

   // run first a data fit 

   const RooArgSet * poiSet = sbModel->GetParametersOfInterest();
   RooRealVar *poi = (RooRealVar*)poiSet->first();

   std::cout << "StandardHypoTestInvDemo : POI initial value:   " << poi->GetName() << " = " << poi->getVal()   << std::endl;  

   // fit the data first (need to use constraint )
   Info( "StandardHypoTestInvDemo"," Doing a first fit to the observed data ");
   if (minimizerType.IsNull()) minimizerType = ROOT::Math::MinimizerOptions::DefaultMinimizerType();
   else 
      ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizerType);
   Info("StandardHypoTestInvDemo","Using %s as minimizer for computing the test statistic",
        ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str() );
   RooArgSet constrainParams;
   if (sbModel->GetNuisanceParameters() ) constrainParams.add(*sbModel->GetNuisanceParameters());
   RooStats::RemoveConstantParameters(&constrainParams);
   TStopwatch tw; 
   tw.Start(); 
   RooFitResult * fitres = sbModel->GetPdf()->fitTo(*data,InitialHesse(false), Hesse(false),
                                                    Minimizer(minimizerType,"Migrad"), Strategy(0), PrintLevel(printLevel+1), Constrain(constrainParams), Save(true) );
   if (fitres->status() != 0) { 
      Warning("StandardHypoTestInvDemo","Fit to the model failed - try with strategy 1 and perform first an Hesse computation");
      fitres = sbModel->GetPdf()->fitTo(*data,InitialHesse(true), Hesse(false),Minimizer(minimizerType,"Migrad"), Strategy(1), PrintLevel(printLevel+1), Constrain(constrainParams), Save(true) );
   }
   if (fitres->status() != 0) 
      Warning("StandardHypoTestInvDemo"," Fit still failed - continue anyway.....");


   double poihat  = poi->getVal();
   std::cout << "StandardHypoTestInvDemo - Best Fit value : " << poi->GetName() << " = "  
             << poihat << " +/- " << poi->getError() << std::endl;
   std::cout << "Time for fitting : "; tw.Print(); 

   //save best fit value in the poi snapshot 
   sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());
   std::cout << "StandardHypoTestInvo: snapshot of S+B Model " << sbModel->GetName() 
             << " is set to the best fit value" << std::endl;

   // build test statistics and hypotest calculators for running the inverter 

   SimpleLikelihoodRatioTestStat slrts(*sbModel->GetPdf(),*bModel->GetPdf());

   if (sbModel->GetSnapshot()) slrts.SetNullParameters(*sbModel->GetSnapshot());
   if (bModel->GetSnapshot()) slrts.SetAltParameters(*bModel->GetSnapshot());

   // ratio of profile likelihood - need to pass snapshot for the alt
   RatioOfProfiledLikelihoodsTestStat 
      ropl(*sbModel->GetPdf(), *bModel->GetPdf(), bModel->GetSnapshot());
   ropl.SetSubtractMLE(false);

   
   ProfileLikelihoodTestStat profll(*sbModel->GetPdf());
   if (testStatType == 3) profll.SetOneSided(1);
   profll.SetMinimizer(minimizerType);
   if (optimize) { 
      profll.SetReuseNLL(true);
      slrts.setReuseNLL(true);
      profll.SetStrategy(0);
      profll.SetPrintLevel(printLevel);
   }

   if (maxPOI > 0) poi->setMax(maxPOI);  // increase limit

   MaxLikelihoodEstimateTestStat maxll(*sbModel->GetPdf(),*poi); 

   // create the HypoTest calculator class 
   HypoTestCalculatorGeneric *  hc = 0;
   if (type == 0) hc = new FrequentistCalculator(*data, *bModel, *sbModel);
   else if (type == 1) hc = new HybridCalculator(*data, *bModel, *sbModel);
   else if (type == 2) hc = new AsymptoticCalculator(*data, *bModel, *sbModel);
   else {
      Error("StandardHypoTestInvDemo","Invalid - calculator type = %d supported values are only :\n\t\t\t 0 (Frequentist) , 1 (Hybrid) , 2 (Asymptotic) ",type);
      return 0;
   }

   // set the test statistic 
   TestStatistic * testStat = 0;
   if (testStatType == 0) testStat = &slrts;
   if (testStatType == 1) testStat = &ropl;
   if (testStatType == 2 || testStatType == 3) testStat = &profll;
   if (testStatType == 4) testStat = &maxll;
   if (testStat == 0) { 
      Error("StandardHypoTestInvDemo","Invalid - test statistic type = %d supported values are only :\n\t\t\t 0 (SLR) , 1 (Tevatron) , 2 (PLR), 3 (PLR1), 4(MLE)",testStatType);
      return 0;
   }
   

   ToyMCSampler *toymcs = (ToyMCSampler*)hc->GetTestStatSampler();
   if (toymcs) { 
      if (useNumberCounting) toymcs->SetNEventsPerToy(1);
      toymcs->SetTestStatistic(testStat);

      if (data->isWeighted() && !generateBinned) { 
         Info("StandardHypoTestInvDemo","Data set is weighted, nentries = %d and sum of weights = %8.1f but toy generation is unbinned - it would be faster to set generateBinned to true\n",data->numEntries(), data->sumEntries());
      }
      toymcs->SetGenerateBinned(generateBinned);

      if (optimize) { 
         // work only of b pdf and s+b pdf are the same
         if (bModel->GetPdf() == sbModel->GetPdf() ) 
            toymcs->SetUseMultiGen(true);
      }

      if (generateBinned &&  sbModel->GetObservables()->getSize() > 2) { 
         Warning("StandardHypoTestInvDemo","generate binned is activated but the number of ovservable is %d. Too much memory could be needed for allocating all the bins",sbModel->GetObservables()->getSize() );
      }

   }


   if (type == 1) { 
      HybridCalculator *hhc = dynamic_cast<HybridCalculator*> (hc);
      assert(hhc);

      hhc->SetToys(ntoys,ntoys/nToysRatio); // can use less ntoys for b hypothesis 

      // remove global observables from ModelConfig (this is probably not needed anymore in 5.32)
      bModel->SetGlobalObservables(RooArgSet() );
      sbModel->SetGlobalObservables(RooArgSet() );


      // check for nuisance prior pdf in case of nuisance parameters 
      if (bModel->GetNuisanceParameters() || sbModel->GetNuisanceParameters() ) {
         RooAbsPdf * nuisPdf = 0; 
         if (nuisPriorName) nuisPdf = w->pdf(nuisPriorName);
         // use prior defined first in bModel (then in SbModel)
         if (!nuisPdf)  { 
            Info("StandardHypoTestInvDemo","No nuisance pdf given for the HybridCalculator - try to deduce  pdf from the model");
            if (bModel->GetPdf() && bModel->GetObservables() ) 
               nuisPdf = RooStats::MakeNuisancePdf(*bModel,"nuisancePdf_bmodel");
            else 
               nuisPdf = RooStats::MakeNuisancePdf(*sbModel,"nuisancePdf_sbmodel");
         }   
         if (!nuisPdf ) {
            if (bModel->GetPriorPdf())  { 
               nuisPdf = bModel->GetPriorPdf();
               Info("StandardHypoTestInvDemo","No nuisance pdf given - try to use %s that is defined as a prior pdf in the B model",nuisPdf->GetName());            
            }
            else { 
               Error("StandardHypoTestInvDemo","Cannnot run Hybrid calculator because no prior on the nuisance parameter is specified or can be derived");
               return 0;
            }
         }
         assert(nuisPdf);
         Info("StandardHypoTestInvDemo","Using as nuisance Pdf ... " );
         nuisPdf->Print();

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
   else if (type == 2) { 
      ((AsymptoticCalculator*) hc)->SetOneSided(true); 
      ((AsymptoticCalculator*) hc)->SetQTilde(true); 
      ((AsymptoticCalculator*) hc)->SetPrintLevel(printLevel+1); 
   }
   else if (type != 2) 
      ((FrequentistCalculator*) hc)->SetToys(ntoys,ntoys/nToysRatio); 

   // Get the result
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);



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
      if (poimin > poimax) { 
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

   tw.Start();
   HypoTestInverterResult * r = calc.GetInterval();
   std::cout << "Time to perform limit scan \n";
   tw.Print();

   if (rebuild) {
      calc.SetCloseProof(1);
      tw.Start();
      SamplingDistribution * limDist = calc.GetUpperLimitDistribution(true,nToyToRebuild);
      std::cout << "Time to rebuild distributions " << std::endl;
      tw.Print();
      
      if (limDist) { 
         std::cout << "expected up limit " << limDist->InverseCDF(0.5) << " +/- " 
                   << limDist->InverseCDF(0.16) << "  " 
                   << limDist->InverseCDF(0.84) << "\n"; 

         //update r to a new updated result object containing the rebuilt expected p-values distributions
         // (it will not recompute the expected limit)
         if (r) delete r;  // need to delete previous object since GetInterval will return a cloned copy
         r = calc.GetInterval();

      }
      else 
         std::cout << "ERROR : failed to re-build distributions " << std::endl; 
   }

   return r; 
}

void ReadResult(const char * fileName, const char * resultName="", bool useCLs=true) { 
   // read a previous stored result from a file given the result name

   StandardHypoTestInvDemo(fileName, resultName,"","","",0,0,useCLs);
}

// int main() {
//    StandardHypoTestInvDemo();
// }
