// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
BEGIN_HTML
<p>
</p>
END_HTML
*/
//



//#define DEBUG
#include "Helper.h"
#include "RooStats/ModelConfig.h"

#include "RooArgSet.h"
#include "RooRealVar.h"

using namespace std; 

namespace RooStats{
namespace HistFactory{
  vector<pair<string, string> > get_comb(vector<string> names){
    vector<pair<string, string> > list;
    for(vector<string>::iterator itr=names.begin(); itr!=names.end(); ++itr){
      vector<string>::iterator itr2=itr; 
      for(itr2++; itr2!=names.end(); ++itr2){
        list.push_back(pair<string, string>(*itr, *itr2));
      }
    }
    return list;
  }

  vector<EstimateSummary>*  loadSavedInputs(TFile* outFile, string channel ){
    outFile->ShowStreamerInfo();

    vector<EstimateSummary>* summaries = new  vector<EstimateSummary>;
    outFile->cd(channel.c_str());

    // loop through estimate summaries
    TIter next(gDirectory->GetListOfKeys()); 
    EstimateSummary* summary; 
    while ((summary=(EstimateSummary*) next())) { 
//      if(summary){
        summary->Print();
        cout << "was able to read summary with name " << summary->name << endl;
        cout << " nominal hist = " << summary->nominal << endl;
        if(summary->nominal)
           cout << " hist name = " << summary->nominal->GetName() <<endl;
        cout << "still ok" << endl;
       
        summaries->push_back(*summary);

//L.M. This code cannot be reached- remove it 
//       }
//       else{
//         cout << "was not able to read summary" << endl;
//       }
    } 
    return summaries;
  }


  void saveInputs(TFile* outFile, string channel, vector<EstimateSummary> summaries){
    vector<EstimateSummary>::iterator it = summaries.begin();
    vector<EstimateSummary>::iterator end = summaries.end();
    vector<TH1*>::iterator histIt;
    vector<TH1*>::iterator histEnd;
    outFile->mkdir(channel.c_str());

    for(; it!=end; ++it){
      if(channel != it->channel){
        cout << "channel mismatch" << endl;
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


  TH1 * GetHisto( TFile * inFile, const string name ){

  if(!inFile || name.empty()){
    cerr << "Not all necessary info are set to access the input file. Check your config" << endl;
    cerr << "fileptr: " << inFile
         << "path/obj: " << name << endl;
    return 0;
  }
  #ifdef DEBUG
    cout << "Retrieving " << name ;
  #endif
    TH1 * ptr = (TH1 *) (inFile->Get( name.c_str() )->Clone());  
  #ifdef DEBUG
    cout << " found at " << ptr << " with integral " << ptr->Integral() << " and mean " << ptr->GetMean() << endl;
  #endif
    if (ptr) ptr->SetDirectory(0); //         for the current histogram h
    //TH1::AddDirectory(kFALSE);
    return ptr;

  }

  TH1 * GetHisto( const string file, const string path, const string obj){

  #ifdef DEBUG
    cout << "Retrieving " << file << ":" << path << obj ;
  #endif
    TFile inFile(file.c_str());
    TH1 * ptr = (TH1 *) (inFile.Get( (path+obj).c_str() )->Clone());
  #ifdef DEBUG
    cout << " found at " << ptr << " with integral " << ptr->Integral() << " and mean " << ptr->GetMean() << endl;
  #endif
    //    if(file.empty() || path.empty() || obj.empty()){
    if(!ptr){
      cerr << "Not all necessary info are set to access the input file. Check your config" << endl;
      cerr << "filename: " << file
           << "path: " << path
           << "obj: " << obj << endl;
    }
    else 
       ptr->SetDirectory(0); //         for the current histogram h

    return ptr;

  }

  void AddSubStrings( vector<string> & vs, string s){
    const string delims("\\ ");
    string::size_type begIdx, endIdx;
    begIdx=s.find_first_not_of(delims);
    while(begIdx!=string::npos){
      endIdx=s.find_first_of(delims, begIdx);
      if(endIdx==string::npos) endIdx=s.length();
      vs.push_back(s.substr(begIdx,endIdx-begIdx));
      begIdx=s.find_first_not_of(delims, endIdx);
    }
  }

  // Turn a string of "children" (space separated items)
  // into a vector of strings
  std::vector<std::string> GetChildrenFromString( std::string str ) {

    std::vector<std::string> child_vec;

    const string delims("\\ ");
    string::size_type begIdx, endIdx;
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

  /*
  bool AddSummaries( vector<EstimateSummary> & channel, vector<vector<EstimateSummary> > &master){
    string channel_str=channel[0].channel;
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



std::vector<EstimateSummary> GetChannelEstimateSummaries(Measurement& measurement, Channel& channel) {

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






}
}
