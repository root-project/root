// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////

/** \class RooStats::HistFactory::Helper
 *  \ingroup HistFactory
 */



//#define DEBUG

#include <sstream>
#include "Helper.h"
#include "RooStats/ModelConfig.h"

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooMinimizer.h"
#include "RooSimultaneous.h"
#include "RooCategory.h"
#include "RooFitResult.h"

using namespace std;

namespace RooStats {
  namespace HistFactory {


    vector< pair<std::string, std::string> > get_comb(vector<std::string> names){
      vector< pair<std::string, std::string> > list;
      for(vector<std::string>::iterator itr=names.begin(); itr!=names.end(); ++itr){
   vector<std::string>::iterator itr2=itr;
   for(++itr2; itr2!=names.end();++itr2){
     list.push_back(pair<std::string, std::string>(*itr, *itr2));
   }
      }
      return list;
    }

    vector<EstimateSummary>*  loadSavedInputs(TFile* outFile, std::string channel ){
      outFile->ShowStreamerInfo();

      vector<EstimateSummary>* summaries = new  vector<EstimateSummary>;
      outFile->cd(channel.c_str());

      // loop through estimate summaries
      TIter next(gDirectory->GetListOfKeys());
      EstimateSummary* summary;
      while ((summary=(EstimateSummary*) next())) {
   //      if(summary){
        summary->Print();
        cout << "was able to read summary with name " << summary->name << std::endl;
        cout << " nominal hist = " << summary->nominal << std::endl;
        if(summary->nominal)
     cout << " hist name = " << summary->nominal->GetName() <<endl;
        cout << "still ok" << std::endl;

        summaries->push_back(*summary);

   //L.M. This code cannot be reached- remove it
   //       }
   //       else{
   //         cout << "was not able to read summary" << std::endl;
   //       }
      }
      return summaries;
    }


    void saveInputs(TFile* outFile, std::string channel, vector<EstimateSummary> summaries){
      vector<EstimateSummary>::iterator it = summaries.begin();
      vector<EstimateSummary>::iterator end = summaries.end();
      vector<TH1*>::iterator histIt;
      vector<TH1*>::iterator histEnd;
      outFile->mkdir(channel.c_str());

      for(; it!=end; ++it){
   if(channel != it->channel){
     cout << "channel mismatch" << std::endl;
     return;
   }
   outFile->cd(channel.c_str());

   // write the EstimateSummary object
   it->Write();

   gDirectory->mkdir(it->name.c_str());
   gDirectory->cd(it->name.c_str());

   it->nominal->Write();

   histIt = it->lowHists.begin();
   histEnd = it->lowHists.end();
   for(; histIt!=histEnd; ++histIt)
     (*histIt)->Write();

   histIt = it->highHists.begin();
   histEnd = it->highHists.end();
   for(; histIt!=histEnd; ++histIt)
     (*histIt)->Write();

      }
    }


    TH1 * GetHisto( TFile * inFile, const std::string name ){

      if(!inFile || name.empty()){
   cerr << "Not all necessary info are set to access the input file. Check your config" << std::endl;
   cerr << "fileptr: " << inFile
        << "path/obj: " << name << std::endl;
   return 0;
      }
#ifdef DEBUG
      cout << "Retrieving " << name ;
#endif
      TH1 * ptr = (TH1 *) (inFile->Get( name.c_str() )->Clone());
#ifdef DEBUG
      cout << " found at " << ptr << " with integral " << ptr->Integral() << " and mean " << ptr->GetMean() << std::endl;
#endif
      if (ptr) ptr->SetDirectory(0); //         for the current histogram h
      //TH1::AddDirectory(kFALSE);
      return ptr;

    }

    TH1 * GetHisto( const std::string file, const std::string path, const std::string obj){

#ifdef DEBUG
      cout << "Retrieving " << file << ":" << path << obj ;
#endif

      TFile inFile(file.c_str());
      TH1 * ptr = (TH1 *) (inFile.Get( (path+obj).c_str() )->Clone());

#ifdef DEBUG
      cout << " found at " << ptr << " with integral " << ptr->Integral()
      << " and mean " << ptr->GetMean() << std::endl;
#endif
      //    if(file.empty() || path.empty() || obj.empty()){

      if(!ptr){
   cerr << "Not all necessary info are set to access the input file. Check your config"
        << std::endl;
   cerr << "filename: " << file
        << "path: " << path
        << "obj: " << obj << std::endl;
      }
      else
   ptr->SetDirectory(0); //         for the current histogram h

      return ptr;

    }

    void AddSubStrings( vector<std::string> & vs, std::string s){
      const std::string delims("\\ ");
      std::string::size_type begIdx, endIdx;
      begIdx=s.find_first_not_of(delims);
      while(begIdx!=string::npos){
   endIdx=s.find_first_of(delims, begIdx);
   if(endIdx==string::npos) endIdx=s.length();
   vs.push_back(s.substr(begIdx,endIdx-begIdx));
   begIdx=s.find_first_not_of(delims, endIdx);
      }
    }

    // Turn a std::string of "children" (space separated items)
    // into a vector of std::strings
    std::vector<std::string> GetChildrenFromString( std::string str ) {

      std::vector<std::string> child_vec;

      const std::string delims("\\ ");
      std::string::size_type begIdx, endIdx;
      begIdx=str.find_first_not_of(delims);
      while(begIdx!=string::npos){
   endIdx=str.find_first_of(delims, begIdx);
   if(endIdx==string::npos) endIdx=str.length();
   std::string child_name = str.substr(begIdx,endIdx-begIdx);
   child_vec.push_back(child_name);
   begIdx=str.find_first_not_of(delims, endIdx);
      }

      return child_vec;
    }

    // Turn a std::string of "children" (space separated items)
    // into a vector of std::strings
    void AddParamsToAsimov( RooStats::HistFactory::Asimov& asimov, std::string str ) {

      // First, split the string into a list
      // each describing a parameter
      std::vector<std::string> string_list = GetChildrenFromString( str );

      // Next, go through each one and split based
      // on the '=' to separate the name from the val
      // and fill the map
      std::map<std::string, double> param_map;

      for( unsigned int i=0; i < string_list.size(); ++i) {

   std::string param = string_list.at(i);
   // Split the string
   size_t eql_location = param.find("=");

   // If there is no '=' deliminator, we only
   // set the variable constant
   if( eql_location==string::npos ) {
     asimov.SetFixedParam(param);
   }
   else {

     std::string param_name = param.substr(0,eql_location);
     double param_val = atof( param.substr(eql_location+1, param.size()).c_str() );

     std::cout << "ASIMOV - Param Name: " << param_name
          << " Param Val: " << param_val << std::endl;
     // Give the params a value AND set them constant
     asimov.SetParamValue(param_name, param_val);
     asimov.SetFixedParam(param_name);
   }

      }

      return;

    }



    /*
      bool AddSummaries( vector<EstimateSummary> & channel, vector<vector<EstimateSummary> > &master){
      std::string channel_str=channel[0].channel;
      for( unsigned int proc=1;  proc < channel.size(); proc++){
      if(channel[proc].channel != channel_str){
      std::cout << "Illegal channel description, should be " << channel_str << " but found " << channel.at(proc).channel << std::endl;
      std::cout << "name " << channel.at(proc).name << std::endl;
      exit(1);
      }
      master.push_back(channel);
      }
      return true;
      }*/


    // Looking to deprecate this function and convert entirely to Measurements
    std::vector<EstimateSummary> GetChannelEstimateSummaries(Measurement& measurement,
                          Channel& channel) {

      // Convert a "Channel" into a list of "Estimate Summaries"
      // This should only be a temporary function, as the
      // EstimateSummary class should be deprecated

      std::vector<EstimateSummary> channel_estimateSummary;

      std::cout << "Processing data: " << std::endl;

      // Add the data
      EstimateSummary data_es;
      data_es.name = "Data";
      data_es.channel = channel.GetName();
      TH1* data_hist = (TH1*) channel.GetData().GetHisto();
      if( data_hist != NULL ) {
   //data_es.nominal = (TH1*) channel.GetData().GetHisto()->Clone();
   data_es.nominal = data_hist;
   channel_estimateSummary.push_back( data_es );
      }

      // Add the samples
      for( unsigned int sampleItr = 0; sampleItr < channel.GetSamples().size(); ++sampleItr ) {

   EstimateSummary sample_es;
   RooStats::HistFactory::Sample& sample = channel.GetSamples().at( sampleItr );

   std::cout << "Processing sample: " << sample.GetName() << std::endl;

   // Define the mapping
   sample_es.name = sample.GetName();
   sample_es.channel = sample.GetChannelName();
   sample_es.nominal = (TH1*) sample.GetHisto()->Clone();

   std::cout << "Checking NormalizeByTheory" << std::endl;

   if( sample.GetNormalizeByTheory() ) {
     sample_es.normName = "" ; // Really bad, confusion convention
   }
   else {
     TString lumiStr;
     lumiStr += measurement.GetLumi();
     lumiStr.ReplaceAll(' ', TString());
     sample_es.normName = lumiStr ;
   }

   std::cout << "Setting the Histo Systs" << std::endl;

   // Set the Histo Systs:
   for( unsigned int histoItr = 0; histoItr < sample.GetHistoSysList().size(); ++histoItr ) {

     RooStats::HistFactory::HistoSys& histoSys = sample.GetHistoSysList().at( histoItr );

     sample_es.systSourceForHist.push_back( histoSys.GetName() );
     sample_es.lowHists.push_back( (TH1*) histoSys.GetHistoLow()->Clone()  );
     sample_es.highHists.push_back( (TH1*) histoSys.GetHistoHigh()->Clone() );

   }

   std::cout << "Setting the NormFactors" << std::endl;

   for( unsigned int normItr = 0; normItr < sample.GetNormFactorList().size(); ++normItr ) {

     RooStats::HistFactory::NormFactor& normFactor = sample.GetNormFactorList().at( normItr );

     EstimateSummary::NormFactor normFactor_es;
     normFactor_es.name = normFactor.GetName();
     normFactor_es.val  = normFactor.GetVal();
     normFactor_es.high = normFactor.GetHigh();
     normFactor_es.low  = normFactor.GetLow();
     normFactor_es.constant = normFactor.GetConst();

     sample_es.normFactor.push_back( normFactor_es );

   }

   std::cout << "Setting the OverallSysList" << std::endl;

   for( unsigned int sysItr = 0; sysItr < sample.GetOverallSysList().size(); ++sysItr ) {

     RooStats::HistFactory::OverallSys& overallSys = sample.GetOverallSysList().at( sysItr );

     std::pair<double, double> DownUpPair( overallSys.GetLow(), overallSys.GetHigh() );
     sample_es.overallSyst[ overallSys.GetName() ]  = DownUpPair; //

   }

   std::cout << "Checking Stat Errors" << std::endl;

   // Do Stat Error
   sample_es.IncludeStatError  = sample.GetStatError().GetActivate();

   // Set the error and error threshold
   sample_es.RelErrorThreshold = channel.GetStatErrorConfig().GetRelErrorThreshold();
   if( sample.GetStatError().GetErrorHist() ) {
     sample_es.relStatError      = (TH1*) sample.GetStatError().GetErrorHist()->Clone();
   }
   else {
     sample_es.relStatError    = NULL;
   }


   // Set the constraint type;
   Constraint::Type type = channel.GetStatErrorConfig().GetConstraintType();

   // Set the default
   sample_es.StatConstraintType = EstimateSummary::Gaussian;

   if( type == Constraint::Gaussian) {
     std::cout << "Using Gaussian StatErrors" << std::endl;
     sample_es.StatConstraintType = EstimateSummary::Gaussian;
   }
   if( type == Constraint::Poisson ) {
     std::cout << "Using Poisson StatErrors" << std::endl;
     sample_es.StatConstraintType = EstimateSummary::Poisson;
   }


   std::cout << "Getting the shape Factor" << std::endl;

   // Get the shape factor
   if( sample.GetShapeFactorList().size() > 0 ) {
     sample_es.shapeFactorName = sample.GetShapeFactorList().at(0).GetName();
   }
   if( sample.GetShapeFactorList().size() > 1 ) {
     std::cout << "Error: Only One Shape Factor currently supported" << std::endl;
     throw hf_exc();
   }


   std::cout << "Setting the ShapeSysts" << std::endl;

   // Get the shape systs:
   for( unsigned int shapeItr=0; shapeItr < sample.GetShapeSysList().size(); ++shapeItr ) {

     RooStats::HistFactory::ShapeSys& shapeSys = sample.GetShapeSysList().at( shapeItr );

     EstimateSummary::ShapeSys shapeSys_es;
     shapeSys_es.name = shapeSys.GetName();
     shapeSys_es.hist = shapeSys.GetErrorHist();

     // Set the constraint type;
     Constraint::Type systype = shapeSys.GetConstraintType();

     // Set the default
     shapeSys_es.constraint = EstimateSummary::Gaussian;

     if( systype == Constraint::Gaussian) {
       shapeSys_es.constraint = EstimateSummary::Gaussian;
     }
     if( systype == Constraint::Poisson ) {
       shapeSys_es.constraint = EstimateSummary::Poisson;
     }

     sample_es.shapeSysts.push_back( shapeSys_es );

   }

   std::cout << "Adding this sample" << std::endl;

   // Push back
   channel_estimateSummary.push_back( sample_es );

      }

      return channel_estimateSummary;

    }



    /*
    void unfoldConstraints(RooArgSet& initial, RooArgSet& final, RooArgSet& obs, RooArgSet& nuis, int& counter) {
      //
      // Loop through an initial set of constraints
      // and create a final list of constraints
      // This is done recursively


      if (counter > 50) {
   cout << "ERROR::Couldn't unfold constraints!" << std::endl;
   cout << "Initial: " << std::endl;
   initial.Print("v");
   cout << std::endl;
   cout << "Final: " << std::endl;
   final.Print("v");
   return;
      }

      TIterator* itr = initial.createIterator();
      RooAbsPdf* pdf;
      while ((pdf = (RooAbsPdf*)itr->Next())) {
   RooArgSet nuis_tmp = nuis;
   RooArgSet constraint_set(*pdf->getAllConstraints(obs, nuis_tmp, false));
   //if (constraint_set.getSize() > 1)
   //{
   string className(pdf->ClassName());
   if (className != "RooGaussian" && className != "RooLognormal" && className != "RooGamma" && className != "RooPoisson" && className != "RooBifurGauss") {
     counter++;
     unfoldConstraints(constraint_set, final, obs, nuis, counter);
   }
   else {
     final.add(*pdf);
   }
      }
      delete itr;
    }
    */
    /*
    RooAbsData* makeAsimovData(ModelConfig* mcInWs, bool doConditional, RooWorkspace* combWS,
                RooAbsPdf* combPdf, RooDataSet* combData, bool b_only, double doMuHat,
                double muVal, bool signalInjection, bool doNuisPro) {
      ////////////////////
      //make asimov data//
      ////////////////////

      // mcInWs -> The ModelCofig for this likelihood
      // doConditional -> Minimize parameters for asimov quantities
      // b_only -> Make the 'background only' asimov dataset, ie mu=0 (set muVal = 0)
      // doNuisPro -> Set all nuisance parameters to '0' and to constant
      //              before minimizing. This should be done with *care*!!
      //              i.e. It should probably be removed as an option.
      // signalInjection -> If true, then do the following:
      //                    Perform the fit with m=0
      //                    After the fit, set the value to mu=muVal
      //                    so that the asimov is created with that value of mu fixed
      // doMuHat -> Set 'mu' to be float during the fit (in the range -10 to 100)
      //            Even if it floats in the fit, it is still set to
      //            'muVal' before the dataset is made (so the only difference
      //            comes from the other parameters that can float simultaneously with mu

      // Defaults:
      // double doMuHat = false
      // double muVal = -999,
      // bool signalInjection = false
      // bool doNuisPro = true

      if( b_only ) muVal = 0.0;

      int _printLevel = 0;

      // If using signal injection or a non-zero mu value,
      // add a suffix showing that value
      std::stringstream muStr;
      if(signalInjection || !b_only) {
   muStr << "_" << muVal;
      }

      // Create the name of the resulting dataset
      std::string dataSetName;
      if(signalInjection) dataSetName = "signalInjection" + muStr.str();
      else dataSetName = "asimovData" + muStr.str();

      // Set the parameter of interest
      // to the 'background' value
      RooRealVar* mu = (RooRealVar*) mcInWs->GetParametersOfInterest()->first();
      if(signalInjection){
   std::cout << "Asimov: Setting " << mu->GetName() << " value to 0 for fit" << std::endl;
   mu->setVal(0);
      }
      else {
   std::cout << "Asimov: Setting " << mu->GetName() << " value to " << muVal << " for fit" << std::endl;
   mu->setVal(muVal);
      }

      // Get necessary info from the ModelConfig
      RooArgSet mc_obs = *mcInWs->GetObservables();
      RooArgSet mc_globs = *mcInWs->GetGlobalObservables();
      RooArgSet mc_nuis = *mcInWs->GetNuisanceParameters();

      // Create a set of constraint terms, which
      // is stored in 'constraint_set'
      // Make some temporary variables and use the
      // unfoldConstrants function to do this.
      RooArgSet constraint_set;
      int counter_tmp = 0;
      RooArgSet mc_nuis_tmp = mc_nuis;
      RooArgSet constraint_set_tmp(*combPdf->getAllConstraints(mc_obs, mc_nuis_tmp, false));
      unfoldConstraints(constraint_set_tmp, constraint_set, mc_obs, mc_nuis_tmp, counter_tmp);

      // Now that we have the constraint terms, we
      // can create the full lists of nuisance parameters
      // and global variables
      RooArgList nui_list("ordered_nuis");
      RooArgList glob_list("ordered_globs");

      TIterator* cIter = constraint_set.createIterator();
      RooAbsArg* arg;
      while ((arg = (RooAbsArg*)cIter->Next())) {
   RooAbsPdf* pdf = (RooAbsPdf*) arg;
   if (!pdf) continue;

   TIterator* nIter = mc_nuis.createIterator();
   RooRealVar* thisNui = NULL;
   RooAbsArg* nui_arg;
   while((nui_arg = (RooAbsArg*)nIter->Next())) {
     if(pdf->dependsOn(*nui_arg)) {
       thisNui = (RooRealVar*) nui_arg;
       break;
     }
   }
   delete nIter;

   // need this in case the observable isn't fundamental.
   // in this case, see which variable is dependent on the nuisance parameter and use that.
   RooArgSet* components = pdf->getComponents();
   components->remove(*pdf);
   if(components->getSize()) {
     TIterator* itr1 = components->createIterator();
     RooAbsArg* arg1;
     while ((arg1 = (RooAbsArg*)itr1->Next())) {
       TIterator* itr2 = components->createIterator();
       RooAbsArg* arg2;
       while ((arg2 = (RooAbsArg*)itr2->Next())) {
         if(arg1 == arg2) continue;
         if(arg2->dependsOn(*arg1)) {
      components->remove(*arg1);
         }
       }
       delete itr2;
     }
     delete itr1;
   }
   if (components->getSize() > 1) {
     std::cout << "ERROR::Couldn't isolate proper nuisance parameter" << std::endl;
     return NULL;
   }
   else if (components->getSize() == 1) {
     thisNui = (RooRealVar*)components->first();
   }

   TIterator* gIter = mc_globs.createIterator();
   RooRealVar* thisGlob = NULL;
   RooAbsArg* glob_arg;
   while ((glob_arg = (RooAbsArg*)gIter->Next()))
     {
       if (pdf->dependsOn(*glob_arg))
         {
      thisGlob = (RooRealVar*)glob_arg;
      break;
         }
     }
   delete gIter;

   if (!thisNui || !thisGlob)
     {
       std::cout << "WARNING::Couldn't find nui or glob for constraint: " << pdf->GetName() << std::endl;
       continue;
     }

   if (_printLevel >= 1) std::cout << "Pairing nui: " << thisNui->GetName() << ", with glob: " << thisGlob->GetName() << ", from constraint: " << pdf->GetName() << std::endl;

   nui_list.add(*thisNui);
   glob_list.add(*thisGlob);

      } // End Loop over Constraint Terms
      delete cIter;

      //save the snapshots of nominal parameters
      combWS->saveSnapshot("nominalGlobs",glob_list);
      combWS->saveSnapshot("nominalNuis", nui_list);

      RooArgSet nuiSet_tmp(nui_list);

      // Interesting line here:
      if(!doMuHat) {
   std::cout << "Asimov: Setting mu constant in fit" << std::endl;
   mu->setConstant(true);
      }
      else {
   std::cout << "Asimov: Letting mu float in fit (muHat)" << std::endl;
   mu->setRange(-10,100);
      }

      // Conditional: "Minimize the parameters"
      if(doConditional) {

   std::cout << "Starting minimization.." << std::endl;

   // Consider removing this option:
   if(!doNuisPro) {
     std::cout << "Asimov: Setting nuisance parameters constant in the fit (ARE YOU SURE YOU WANT THIS)" << std::endl;
     TIterator* nIter = nuiSet_tmp.createIterator();
     RooRealVar* thisNui = NULL;
     while((thisNui = (RooRealVar*) nIter->Next())) {
       thisNui->setVal(0);
       thisNui->setConstant();
     }
     delete nIter;
     // This should be checked, we don't want to
     if(combWS->var("Lumi")) {
       combWS->var("Lumi")->setVal(1);
       combWS->var("Lumi")->setConstant();
     }
   }

   // Create the nll and its minimizer
   RooAbsReal* nll = combPdf->createNLL(*combData, RooFit::Constrain(nuiSet_tmp));
   RooMinimizer minim(*nll);
   minim.setStrategy(2);
   minim.setPrintLevel(999);

   // Do the minimization
   std::cout << "Minimizing to make Asimov dataset:" << std::endl;
   int status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), "migrad");
   if (status == 0) {
     // status==0 means fit was successful
   std::cout << "Successfully minimized to make Asimov dataset:" << std::endl;
   RooFitResult* fit_result = minim.lastMinuitFit();
   std::cout << "Asimov: Final Fitted Parameters" << std::endl;
   fit_result->Print("V");
   } else{
     std::cout << "Fit failed for mu = " << mu->getVal() << " with status " << status << std::endl;
     std::cout << "Trying minuit..." << std::endl;
     status = minim.minimize("Minuit", "migrad");
     if (status != 0) {
       cout << "Fit failed for mu = " << mu->getVal() << " with status " << status << std::endl;
       throw hf_exc();
     }
   }

   // Undo the 'doNuisPro' part
   // Again, may want to remove this
   if (!doNuisPro) {
     TIterator* nIter = nuiSet_tmp.createIterator();
     RooRealVar* thisNui = NULL;
     while ((thisNui = (RooRealVar*)nIter->Next())) {
       thisNui->setConstant(false);
     }
     delete nIter;
     if (combWS->var("Lumi")) {
       combWS->var("Lumi")->setConstant(false);
     }
   }

   std::cout << "Done" << std::endl;
      } // END: DoConditional

      mu->setConstant(false);

      //loop over the nui/glob list, grab the corresponding variable from the tmp ws,
      // and set the glob to the value of the nui
      int nrNuis = nui_list.getSize();
      if (nrNuis != glob_list.getSize()) {
   std::cout << "ERROR::nui_list.getSize() != glob_list.getSize()!" << std::endl;
   return NULL;
      }

      for(int i=0; i<nrNuis; i++) {
   RooRealVar* nui = (RooRealVar*) nui_list.at(i);
   RooRealVar* glob = (RooRealVar*) glob_list.at(i);

   if (_printLevel >= 1) std::cout << "nui: " << nui << ", glob: " << glob << std::endl;
   if (_printLevel >= 1) std::cout << "Setting glob: " << glob->GetName() << ", which had previous val: " << glob->getVal() << ", to conditional val: " << nui->getVal() << std::endl;

   glob->setVal(nui->getVal());
      }

      //save the snapshots of conditional parameters
      //cout << "Saving conditional snapshots" << std::endl;
      combWS->saveSnapshot(("conditionalGlobs"+muStr.str()).c_str(),glob_list);
      combWS->saveSnapshot(("conditionalNuis" +muStr.str()).c_str(), nui_list);

      if(!doConditional) {
   combWS->loadSnapshot("nominalGlobs");
   combWS->loadSnapshot("nominalNuis");
      }

      //cout << "Making asimov" << std::endl;
      //make the asimov data (snipped from Kyle)

      // Restore the value of mu to the target value
      mu->setVal(muVal);

      ModelConfig* mc = mcInWs;

      int iFrame=0;

      const char* weightName = "weightVar";
      RooArgSet obsAndWeight;
      obsAndWeight.add(*mc->GetObservables());

      // Get the weightVar, or create one if necessary
      RooRealVar* weightVar = combWS->var(weightName); // NULL;
      //  if (!(weightVar = combWS->var(weightName)))
      if( weightVar==NULL ) {
   combWS->import(*(new RooRealVar(weightName, weightName, 1,0,1000)));
   weightVar = combWS->var(weightName);
      }
      if (_printLevel >= 1) std::cout << "weightVar: " << weightVar << std::endl;
      obsAndWeight.add(*combWS->var(weightName));

      //////////////////////////////////////////////////////
      //////////////////////////////////////////////////////
      //////////////////////////////////////////////////////
      //////////////////////////////////////////////////////
      //////////////////////////////////////////////////////
      // MAKE ASIMOV DATA FOR OBSERVABLES

      // dummy var can just have one bin since it's a dummy
      if(combWS->var("ATLAS_dummyX"))  combWS->var("ATLAS_dummyX")->setBins(1);

      if (_printLevel >= 1) std::cout << " check expectedData by category" << std::endl;
      RooSimultaneous* simPdf = dynamic_cast<RooSimultaneous*>(mc->GetPdf());

      // Create the pointer to be returned
      RooDataSet* asimovData=NULL;

      // If the pdf isn't simultaneous:
      if(!simPdf) {

   // Get pdf associated with state from simpdf
   RooAbsPdf* pdftmp = mc->GetPdf();//simPdf->getPdf(channelCat->getLabel()) ;

   // Generate observables defined by the pdf associated with this state
   RooArgSet* obstmp = pdftmp->getObservables(*mc->GetObservables()) ;

   if (_printLevel >= 1) {
     obstmp->Print();
   }


   asimovData = new RooDataSet(dataSetName.c_str(), dataSetName.c_str(),
                RooArgSet(obsAndWeight), RooFit::WeightVar(*weightVar));

   RooRealVar* thisObs = ((RooRealVar*)obstmp->first());
   double expectedEvents = pdftmp->expectedEvents(*obstmp);
   double thisNorm = 0;
   for(int jj=0; jj<thisObs->numBins(); ++jj){
     thisObs->setBin(jj);

     thisNorm=pdftmp->getVal(obstmp)*thisObs->getBinWidth(jj);
     if (thisNorm*expectedEvents <= 0)
       {
         cout << "WARNING::Detected bin with zero expected events (" << thisNorm*expectedEvents << ") ! Please check your inputs. Obs = " << thisObs->GetName() << ", bin = " << jj << std::endl;
       }
     if (thisNorm*expectedEvents > 0 && thisNorm*expectedEvents < pow(10.0, 18)) asimovData->add(*mc->GetObservables(), thisNorm*expectedEvents);
   }

   if (_printLevel >= 1)
     {
       asimovData->Print();
       std::cout <<"sum entries "<<asimovData->sumEntries()<<endl;
     }
   if(asimovData->sumEntries()!=asimovData->sumEntries()){
     std::cout << "sum entries is nan"<<endl;
     exit(1);
   }

   // don't import, return (of course)
   //combWS->import(*asimovData);
      }
      else
   {

     // If it IS a simultaneous pdf

     std::cout << "found a simPdf: " << simPdf << std::endl;
     map<std::string, RooDataSet*> asimovDataMap;

     RooCategory* channelCat = (RooCategory*)&simPdf->indexCat();
     TIterator* iter = channelCat->typeIterator() ;
     RooCatType* tt = NULL;
     int nrIndices = 0;
     while((tt=(RooCatType*) iter->Next())) {
       nrIndices++;
     }

     for (int i=0;i<nrIndices;i++){

       channelCat->setIndex(i);

       std::cout << "Checking channel: " << channelCat->getLabel() << std::endl;
       iFrame++;
       // Get pdf associated with state from simpdf
       RooAbsPdf* pdftmp = simPdf->getPdf(channelCat->getLabel()) ;

       // Generate observables defined by the pdf associated with this state
       RooArgSet* obstmp = pdftmp->getObservables(*mc->GetObservables()) ;

       if (_printLevel >= 1) {
         obstmp->Print();
         cout << "on type " << channelCat->getLabel() << " " << iFrame << std::endl;
       }

       RooDataSet* obsDataUnbinned = new RooDataSet(Form("combAsimovData%d",iFrame),Form("combAsimovData%d",iFrame),
                      RooArgSet(obsAndWeight,*channelCat), RooFit::WeightVar(*weightVar));
       RooRealVar* thisObs = ((RooRealVar*)obstmp->first());
       double expectedEvents = pdftmp->expectedEvents(*obstmp);
       double thisNorm = 0;
       TString pdftmp_name = pdftmp->GetName();

       if (!expectedEvents) {
         std::cout << "Not expected events" << std::endl;
         if (pdftmp_name == "model_E")
      ((RooRealVar*)obstmp->first())->setVal(combWS->function("p_e")->getVal());

         else if (pdftmp_name == "model_MU")
      ((RooRealVar*)obstmp->first())->setVal(combWS->function("p_mu")->getVal());

         else if ((pdftmp_name == "model_ratio_ELMU") || (pdftmp_name == "model_comb")) {
      //((RooRealVar*)obstmp->first())->setVal(combWS->function("p_comb")->getVal());
      double p_asimov_val = combWS->var("p_asimov")->getVal();
      std::cout << "p_asimov val: " << p_asimov_val << std::endl;
      ((RooRealVar*)obstmp->first())->setVal(combWS->var("p_asimov")->getVal());
         }

         else {
      std::cout << "Failed to set asimov data for non-extended pdf" << std::endl;
      exit(1);
         }
         obsDataUnbinned->add(*mc->GetObservables());

       }
       else {
         std::cout << "expected events" << std::endl;
         for(int jj=0; jj<thisObs->numBins(); ++jj){
      thisObs->setBin(jj);

      thisNorm=pdftmp->getVal(obstmp)*thisObs->getBinWidth(jj);
      if (thisNorm*expectedEvents <= 0)
        {
          std::cout << "WARNING::Detected bin with zero expected events (" << thisNorm*expectedEvents << ") ! Please check your inputs. Obs = " << thisObs->GetName() << ", bin = " << jj << std::endl;
        }
      if (thisNorm*expectedEvents > pow(10.0, -9) && thisNorm*expectedEvents < pow(10.0, 9)) obsDataUnbinned->add(*mc->GetObservables(), thisNorm*expectedEvents);
         }
       }

       if (_printLevel >= 1)
         {
      obsDataUnbinned->Print();
      std::cout <<"sum entries "<<obsDataUnbinned->sumEntries()<<endl;
         }
       if(obsDataUnbinned->sumEntries()!=obsDataUnbinned->sumEntries()){
         cout << "sum entries is nan"<<endl;
         exit(1);
       }

       asimovDataMap[string(channelCat->getLabel())] = obsDataUnbinned;//tempData;

       if (_printLevel >= 1)
         {
      std::cout << "channel: " << channelCat->getLabel() << ", data: ";
      obsDataUnbinned->Print();
      std::cout << std::endl;
         }
     }

     channelCat->setIndex(0);

     asimovData = new RooDataSet(dataSetName.c_str(),dataSetName.c_str(),
                  RooArgSet(obsAndWeight,*channelCat), RooFit::Index(*channelCat),
                  RooFit::Import(asimovDataMap), RooFit::WeightVar(*weightVar));

     // Don't import, return (of course)
     //combWS->import(*asimovData);
   } // End if over simultaneous pdf

      combWS->loadSnapshot("nominalNuis");
      combWS->loadSnapshot("nominalGlobs");

      return asimovData;

    }
    */
  }
}
