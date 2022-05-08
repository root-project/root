// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/HistFactory/MakeModelAndMeasurementsFast.h"

// from std
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>

// from root
#include "TFile.h"
#include "TH1F.h"
#include "TString.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TSystem.h"


// from roofit
#include "RooStats/ModelConfig.h"

// from this package
#include "RooStats/HistFactory/EstimateSummary.h"
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/HistFactoryException.h"

#include "HFMsgService.h"

using namespace RooFit;
//using namespace RooStats;
//using namespace HistFactory;

//using namespace std;


/** ********************************************************************************************
  \ingroup HistFactory

  <p>
  This is a package that creates a RooFit probability density function from ROOT histograms
  of expected distributions and histograms that represent the +/- 1 sigma variations
  from systematic effects. The resulting probability density function can then be used
  with any of the statistical tools provided within RooStats, such as the profile
  likelihood ratio, Feldman-Cousins, etc.  In this version, the model is directly
  fed to a likelihood ratio test, but it needs to be further factorized.</p>

  <p>
  The user needs to provide histograms (in picobarns per bin) and configure the job
  with XML.  The configuration XML is defined in the file `$ROOTSYS/config/HistFactorySchema.dtd`, but essentially
  it is organized as follows (see the examples in `${ROOTSYS}/tutorials/histfactory/`)</p>

  <ul>
  <li> a top level 'Combination' that is composed of:</li>
  <ul>
  <li> several 'Channels' (eg. ee, emu, mumu), which are composed of:</li>
  <ul>
  <li> several 'Samples' (eg. signal, bkg1, bkg2, ...), each of which has:</li>
  <ul>
  <li> a name</li>
  <li> if the sample is normalized by theory (eg N = L*sigma) or not (eg. data driven)</li>
  <li> a nominal expectation histogram</li>
  <li> a named 'Normalization Factor' (which can be fixed or allowed to float in a fit)</li>
  <li> several 'Overall Systematics' in normalization with:</li>
  <ul>
  <li> a name</li>
  <li> +/- 1 sigma variations (eg. 1.05 and 0.95 for a 5% uncertainty)</li>
  </ul>
  <li> several 'Histogram Systematics' in shape with:</li>
  <ul>
  <li> a name (which can be shared with the OverallSyst if correlated)</li>
  <li> +/- 1 sigma variational histograms</li>
  </ul>
  </ul>
  </ul>
  <li> several 'Measurements' (corresponding to a full fit of the model) each of which specifies</li>
  <ul>
  <li> a name for this fit to be used in tables and files</li>
  <li> what is the luminosity associated to the measurement in picobarns</li>
  <li> which bins of the histogram should be used</li>
  <li> what is the relative uncertainty on the luminosity </li>
  <li> what is (are) the parameter(s) of interest that will be measured</li>
  <li> which parameters should be fixed/floating (eg. nuisance parameters)</li>
  </ul>
  </ul>
  </ul>
*/
RooWorkspace* RooStats::HistFactory::MakeModelAndMeasurementFast( RooStats::HistFactory::Measurement& measurement, HistoToWorkspaceFactoryFast::Configuration const& cfg)
{
  // This will be returned
  RooWorkspace* ws = NULL;
  TFile* outFile = NULL;
  FILE*  tableFile=NULL;

  auto& msgSvc = RooMsgService::instance();
  msgSvc.getStream(1).removeTopic(RooFit::ObjectHandling);

  try {

    cxcoutIHF << "Making Model and Measurements (Fast) for measurement: " << measurement.GetName() << std::endl;

    double lumiError = measurement.GetLumi()*measurement.GetLumiRelErr();

    cxcoutIHF << "using lumi = " << measurement.GetLumi() << " and lumiError = " << lumiError
         << " including bins between " << measurement.GetBinLow() << " and " << measurement.GetBinHigh() << std::endl;

    std::ostringstream parameterMessage;
    parameterMessage << "fixing the following parameters:"  << std::endl;

    for(std::vector<std::string>::iterator itr=measurement.GetConstantParams().begin(); itr!=measurement.GetConstantParams().end(); ++itr){
      parameterMessage << "   " << *itr << '\n';
    }
    cxcoutIHF << parameterMessage.str();

    std::string rowTitle = measurement.GetName();

    std::vector<std::unique_ptr<RooWorkspace>> channel_workspaces;
    std::vector<std::string>        channel_names;

    // Create the outFile - first check if the outputfile exists
    std::string prefix =  measurement.GetOutputFilePrefix();
    // parse prefix to find output directory -
    // assume there is a file prefix after the last "/" that we remove
    // to get the directory name.
    // We do by finding last occurrence of "/" and using as directory name what is before
    // if we do not have a "/" in the prefix there is no output directory to be checked or created
    size_t pos = prefix.rfind("/");
    if (pos != std::string::npos) {
       std::string outputDir = prefix.substr(0,pos);
       cxcoutDHF << "Checking if output directory : " << outputDir << " -  exists" << std::endl;
       if (gSystem->OpenDirectory( outputDir.c_str() )  == 0 ) {
          cxcoutDHF << "Output directory : " << outputDir << " - does not exist, try to create" << std::endl;
          int success = gSystem->MakeDirectory( outputDir.c_str() );
          if( success != 0 ) {
             std::string fullOutputDir = std::string(gSystem->pwd()) + std::string("/") + outputDir;
             cxcoutEHF << "Error: Failed to make output directory: " <<  fullOutputDir << std::endl;
             throw hf_exc();
          }
       }
    }

    // This holds the TGraphs that are created during the fit
    std::string outputFileName = measurement.GetOutputFilePrefix() + "_" + measurement.GetName() + ".root";
    cxcoutIHF << "Creating the output file: " << outputFileName << std::endl;
    outFile = new TFile(outputFileName.c_str(), "recreate");

    // Create the table file
    // This holds the table of fitted values and errors
    std::string tableFileName = measurement.GetOutputFilePrefix() + "_results.table";
    cxcoutIHF << "Creating the table file: " << tableFileName << std::endl;
    tableFile =  fopen( tableFileName.c_str(), "a");

    cxcoutIHF << "Creating the HistoToWorkspaceFactoryFast factory" << std::endl;
    HistoToWorkspaceFactoryFast factory{measurement, cfg};

    // Make the factory, and do some preprocessing
    // HistoToWorkspaceFactoryFast factory(measurement, rowTitle, outFile);
    cxcoutIHF << "Setting preprocess functions" << std::endl;
    factory.SetFunctionsToPreprocess( measurement.GetPreprocessFunctions() );

    // for results tables
    fprintf(tableFile, " %s &", rowTitle.c_str() );

    // First: Loop to make the individual channels
    for( unsigned int chanItr = 0; chanItr < measurement.GetChannels().size(); ++chanItr ) {

      HistFactory::Channel& channel = measurement.GetChannels().at( chanItr );
      if( ! channel.CheckHistograms() ) {
   cxcoutEHF << "MakeModelAndMeasurementsFast: Channel: " << channel.GetName()
        << " has uninitialized histogram pointers" << std::endl;
   throw hf_exc();
      }

      // Make the workspace for this individual channel
      std::string ch_name = channel.GetName();
      cxcoutPHF << "Starting to process channel: " << ch_name << std::endl;
      channel_names.push_back(ch_name);
      RooWorkspace* ws_single = factory.MakeSingleChannelModel( measurement, channel );
      channel_workspaces.emplace_back(ws_single);

      // Make the output
      std::string ChannelFileName = measurement.GetOutputFilePrefix() + "_"
   + ch_name + "_" + rowTitle + "_model.root";
      ws_single->writeToFile( ChannelFileName.c_str() );

      // Now, write the measurement to the file
      // Make a new measurement for only this channel
      RooStats::HistFactory::Measurement meas_chan( measurement );
      meas_chan.GetChannels().clear();
      meas_chan.GetChannels().push_back( channel );
      cxcoutIHF << "Opening File to hold channel: " << ChannelFileName << std::endl;
      TFile* chanFile = TFile::Open( ChannelFileName.c_str(), "UPDATE" );
      cxcoutIHF << "About to write channel measurement to file" << std::endl;
      meas_chan.writeToFile( chanFile );
      cxcoutPHF << "Successfully wrote channel to file" << std::endl;
      chanFile->Close();

      // Get the Paramater of Interest as a RooRealVar
      RooRealVar* poi = dynamic_cast<RooRealVar*>( ws_single->var(measurement.GetPOI()));

      // do fit unless exportOnly requested
      if(! measurement.GetExportOnly()){
   if(!poi) {
     cxcoutWHF << "Can't do fit for: " << measurement.GetName()
          << ", no parameter of interest" << std::endl;
   } else {
     if(ws_single->data("obsData")) {
       FitModelAndPlot(measurement.GetName(), measurement.GetOutputFilePrefix(), ws_single,
             ch_name, "obsData",    outFile, tableFile);
     } else {
       FitModelAndPlot(measurement.GetName(), measurement.GetOutputFilePrefix(), ws_single,
             ch_name, "asimovData", outFile, tableFile);
     }
   }
      }

      fprintf(tableFile, " & " );
    } // End loop over channels

    /***
   Second: Make the combined model:
   If you want output histograms in root format, create and pass it to the combine routine.
   "combine" : will do the individual cross-section measurements plus combination
    ***/

    // Use HistFactory to combine the individual channel workspaces
    ws = factory.MakeCombinedModel(channel_names, channel_workspaces);

    // Configure that workspace
    HistoToWorkspaceFactoryFast::ConfigureWorkspaceForMeasurement( "simPdf", ws, measurement );

    // Get the Parameter of interest as a RooRealVar
    RooRealVar* poi = dynamic_cast<RooRealVar*>( ws->var(measurement.GetPOI()));

    std::string CombinedFileName = measurement.GetOutputFilePrefix() + "_combined_"
      + rowTitle + "_model.root";
    cxcoutPHF << "Writing combined workspace to file: " << CombinedFileName << std::endl;
    ws->writeToFile( CombinedFileName.c_str() );
    cxcoutPHF << "Writing combined measurement to file: " << CombinedFileName << std::endl;
    TFile* combFile = TFile::Open( CombinedFileName.c_str(), "UPDATE" );
    if( combFile == NULL ) {
      cxcoutEHF << "Error: Failed to open file " << CombinedFileName << std::endl;
      throw hf_exc();
    }
    measurement.writeToFile( combFile );
    combFile->Close();

    // Fit the combined model
    if(! measurement.GetExportOnly()){
      if(!poi) {
   cxcoutWHF << "Can't do fit for: " << measurement.GetName()
        << ", no parameter of interest" << std::endl;
      }
      else {
   if(ws->data("obsData")){
     FitModelAndPlot(measurement.GetName(), measurement.GetOutputFilePrefix(), ws,"combined",
           "obsData",    outFile, tableFile);
   }
   else {
     FitModelAndPlot(measurement.GetName(), measurement.GetOutputFilePrefix(), ws,"combined",
           "asimovData", outFile, tableFile);
   }
      }
    }

    fprintf(tableFile, " \\\\ \n");

    outFile->Close();
    delete outFile;

    fclose( tableFile );

  }
  catch(...) {
    if( tableFile ) fclose(tableFile);
    if(outFile) outFile->Close();
    throw;
  }

  msgSvc.getStream(1).addTopic(RooFit::ObjectHandling);

  return ws;

}


///////////////////////////////////////////////
void RooStats::HistFactory::FitModelAndPlot(const std::string& MeasurementName,
                   const std::string& FileNamePrefix,
                   RooWorkspace * combined, std::string channel,
                   std::string data_name,
                   TFile* outFile, FILE* tableFile  ) {

  if( outFile == NULL ) {
    cxcoutEHF << "Error: Output File in FitModelAndPlot is NULL" << std::endl;
    throw hf_exc();
  }

  if( tableFile == NULL ) {
    cxcoutEHF << "Error: tableFile in FitModelAndPlot is NULL" << std::endl;
    throw hf_exc();
  }

  if( combined == NULL ) {
    cxcoutEHF << "Error: Supplied workspace in FitModelAndPlot is NULL" << std::endl;
    throw hf_exc();
  }

  ModelConfig* combined_config = (ModelConfig *) combined->obj("ModelConfig");
  if(!combined_config){
    cxcoutEHF << "Error: no ModelConfig found in Measurement: "
         << MeasurementName <<  std::endl;
    throw hf_exc();
  }

  RooAbsData* simData = combined->data(data_name.c_str());
  if(!simData){
    cxcoutEHF << "Error: Failed to get dataset: " << data_name
         << " in measurement: " << MeasurementName << std::endl;
    throw hf_exc();
  }

  const RooArgSet* POIs = combined_config->GetParametersOfInterest();
  if(!POIs) {
    cxcoutEHF << "Not Fitting Model for measurement: " << MeasurementName
         << ", no poi found" << std::endl;
    // Should I throw an exception here?
    return;
  }

  RooAbsPdf* model = combined_config->GetPdf();
  if( model==NULL ) {
    cxcoutEHF << "Error: Failed to find pdf in ModelConfig: " << combined_config->GetName()
         << std::endl;
    throw hf_exc();
  }

  // Save a Snapshot
  RooArgSet PoiPlusNuisance;
  if( combined_config->GetNuisanceParameters() ) {
    PoiPlusNuisance.add( *combined_config->GetNuisanceParameters() );
  }
  PoiPlusNuisance.add( *combined_config->GetParametersOfInterest() );
  combined->saveSnapshot("InitialValues", PoiPlusNuisance);

  ///////////////////////////////////////
  // Do the fit
  cxcoutPHF << "\n---------------"
    << "\nDoing "<< channel << " Fit"
    << "\n---------------\n\n" << std::endl;
  model->fitTo(*simData, Minos(true), PrintLevel(RooMsgService::instance().isActive(static_cast<TObject*>(nullptr), RooFit::HistFactory, RooFit::DEBUG) ? 1 : -1));

  // If there are no parameters of interest,
  // we exit the function here
  if( POIs->empty() ) {
    cxcoutWHF << "WARNING: No POIs found in measurement: " << MeasurementName << std::endl;
    return;
  }

  // Loop over all POIs and print their fitted values
  for (auto const *poi : static_range_cast<RooRealVar *>(*POIs)) {
    cxcoutIHF << "printing results for " << poi->GetName()
         << " at " << poi->getVal()<< " high "
         << poi->getErrorLo() << " low "
         << poi->getErrorHi() << std::endl;
  }

  // But we only make detailed plots and tables
  // for the 'first' POI
  RooRealVar* poi = static_cast<RooRealVar *>(POIs->first());

  // Print the MINOS errors to the TableFile
  fprintf(tableFile, " %.4f / %.4f  ", poi->getErrorLo(), poi->getErrorHi());

  // Make the Profile Likelihood Plot
  RooAbsReal* nll = model->createNLL(*simData);
  RooAbsReal* profile = nll->createProfile(*poi);
  if( profile==NULL ) {
    cxcoutEHF << "Error: Failed to make ProfileLikelihood for: " << poi->GetName()
         << " using model: " << model->GetName()
         << " and data: " << simData->GetName()
         << std::endl;
    throw hf_exc();
  }

  RooPlot* frame = poi->frame();
  if( frame == NULL ) {
    cxcoutEHF << "Error: Failed to create RooPlot frame for: " << poi->GetName() << std::endl;
    throw hf_exc();
  }

  // Draw the likelihood curve
  FormatFrameForLikelihood(frame);
  TCanvas* ProfileLikelihoodCanvas = new TCanvas( channel.c_str(), "",800,600);
  nll->plotOn(frame, ShiftToZero(), LineColor(kRed), LineStyle(kDashed));
  profile->plotOn(frame);
  frame->SetMinimum(0);
  frame->SetMaximum(2.);
  frame->Draw();
  std::string ProfilePlotName = FileNamePrefix+"_"+channel+"_"+MeasurementName+"_profileLR.eps";
  ProfileLikelihoodCanvas->SaveAs( ProfilePlotName.c_str() );
  delete ProfileLikelihoodCanvas;

  // Now, we save our results to the 'output' file
  // (I'm not sure if users actually look into this file,
  // but adding additional information and useful plots
  // may make it more attractive)

  // Save to the output file
  TDirectory* channel_dir = outFile->mkdir(channel.c_str());
  if( channel_dir == NULL ) {
    cxcoutEHF << "Error: Failed to make channel directory: " << channel << std::endl;
    throw hf_exc();
  }
  TDirectory* summary_dir = channel_dir->mkdir("Summary");
  if( summary_dir == NULL ) {
    cxcoutEHF << "Error: Failed to make Summary directory for channel: "
         << channel << std::endl;
    throw hf_exc();
  }
  summary_dir->cd();

  // Save a graph of the profile likelihood curve
  RooCurve* curve=frame->getCurve();
  Int_t curve_N=curve->GetN();
  double* curve_x=curve->GetX();

  double * x_arr = new double[curve_N];
  double * y_arr_nll = new double[curve_N];

  for(int i=0; i<curve_N; i++){
    double f=curve_x[i];
    poi->setVal(f);
    x_arr[i]=f;
    y_arr_nll[i]=nll->getVal();
  }

  delete frame;

  TGraph* g = new TGraph(curve_N, x_arr, y_arr_nll);
  g->SetName( (FileNamePrefix +"_nll").c_str() );
  g->Write();
  delete g;
  delete [] x_arr;
  delete [] y_arr_nll;

  // Finally, restore the initial values
  combined->loadSnapshot("InitialValues");

}


void RooStats::HistFactory::FitModel(RooWorkspace * combined, std::string data_name ) {

    cxcoutIHF << "In Fit Model" << std::endl;
    ModelConfig * combined_config = (ModelConfig *) combined->obj("ModelConfig");
    if(!combined_config){
      cxcoutEHF << "no model config " << "ModelConfig" << " exiting" << std::endl;
      return;
    }

    RooAbsData* simData = combined->data(data_name.c_str());
    if(!simData){
      cxcoutEHF << "no data " << data_name << " exiting" << std::endl;
      return;
    }

    const RooArgSet * POIs=combined_config->GetParametersOfInterest();
    if(!POIs){
      cxcoutEHF << "no poi " << data_name << " exiting" << std::endl;
      return;
    }

    RooAbsPdf* model=combined_config->GetPdf();
    model->fitTo(*simData, Minos(true), PrintLevel(1));

  }


void RooStats::HistFactory::FormatFrameForLikelihood(RooPlot* frame, std::string /*XTitle*/,
                       std::string YTitle){

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    // gStyle->SetPadColor(0);
    // gStyle->SetCanvasColor(255);
    // gStyle->SetTitleFillColor(255);
    // gStyle->SetFrameFillColor(0);
    // gStyle->SetStatColor(255);

    RooAbsRealLValue* var = frame->getPlotVar();
    double xmin = var->getMin();
    double xmax = var->getMax();

    frame->SetTitle("");
    //      frame->GetXaxis()->SetTitle(XTitle.c_str());
    frame->GetXaxis()->SetTitle(var->GetTitle());
    frame->GetYaxis()->SetTitle(YTitle.c_str());
    frame->SetMaximum(2.);
    frame->SetMinimum(0.);
    TLine * line = new TLine(xmin,.5,xmax,.5);
    line->SetLineColor(kGreen);
    TLine * line90 = new TLine(xmin,2.71/2.,xmax,2.71/2.);
    line90->SetLineColor(kGreen);
    TLine * line95 = new TLine(xmin,3.84/2.,xmax,3.84/2.);
    line95->SetLineColor(kGreen);
    frame->addObject(line);
    frame->addObject(line90);
    frame->addObject(line95);
}


// Looking to deprecate this function and convert entirely to Measurements
std::vector<RooStats::HistFactory::EstimateSummary> RooStats::HistFactory::GetChannelEstimateSummaries(
        RooStats::HistFactory::Measurement& measurement,
        RooStats::HistFactory::Channel& channel) {

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
