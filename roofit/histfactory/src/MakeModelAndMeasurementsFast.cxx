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

// from roofit
#include "RooFit/ModelConfig.h"

// from this package
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/HistFactoryException.h"

#include "HFMsgService.h"

#include <TFile.h>
#include <TSystem.h>

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>

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
RooFit::OwningPtr<RooWorkspace>
RooStats::HistFactory::MakeModelAndMeasurementFast(RooStats::HistFactory::Measurement &measurement,
                                                   HistoToWorkspaceFactoryFast::Configuration const &cfg)
{
  std::unique_ptr<TFile> outFile;

  auto& msgSvc = RooMsgService::instance();
  msgSvc.getStream(1).removeTopic(RooFit::ObjectHandling);

    cxcoutIHF << "Making Model and Measurements (Fast) for measurement: " << measurement.GetName() << std::endl;

    double lumiError = measurement.GetLumi()*measurement.GetLumiRelErr();

    cxcoutIHF << "using lumi = " << measurement.GetLumi() << " and lumiError = " << lumiError
         << " including bins between " << measurement.GetBinLow() << " and " << measurement.GetBinHigh() << std::endl;

    std::ostringstream parameterMessage;
    parameterMessage << "fixing the following parameters:"  << std::endl;

    for (auto const &name : measurement.GetConstantParams()) {
      parameterMessage << "   " << name << '\n';
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
    size_t pos = prefix.rfind('/');
    if (pos != std::string::npos) {
       std::string outputDir = prefix.substr(0,pos);
       cxcoutDHF << "Checking if output directory : " << outputDir << " -  exists" << std::endl;
       if (gSystem->OpenDirectory( outputDir.c_str() )  == nullptr ) {
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
    outFile = std::make_unique<TFile>(outputFileName.c_str(), "recreate");

    cxcoutIHF << "Creating the HistoToWorkspaceFactoryFast factory" << std::endl;
    HistoToWorkspaceFactoryFast factory{measurement, cfg};

    // Make the factory, and do some preprocessing
    // HistoToWorkspaceFactoryFast factory(measurement, rowTitle, outFile);
    cxcoutIHF << "Setting preprocess functions" << std::endl;
    factory.SetFunctionsToPreprocess( measurement.GetPreprocessFunctions() );

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
      std::unique_ptr<RooWorkspace> ws_single{factory.MakeSingleChannelModel( measurement, channel )};

      if (cfg.createPerRegionWorkspaces) {
        // Make the output
        std::string ChannelFileName = measurement.GetOutputFilePrefix() + "_" 
          + ch_name + "_" + rowTitle + "_model.root";
        cxcoutIHF << "Opening File to hold channel: " << ChannelFileName << std::endl;
        std::unique_ptr<TFile> chanFile{TFile::Open( ChannelFileName.c_str(), "RECREATE" )};
        chanFile->WriteTObject(ws_single.get());
        // Now, write the measurement to the file
        // Make a new measurement for only this channel
        RooStats::HistFactory::Measurement meas_chan( measurement );
        meas_chan.GetChannels().clear();
        meas_chan.GetChannels().push_back( channel );
        cxcoutIHF << "About to write channel measurement to file" << std::endl;
        meas_chan.writeToFile( chanFile.get() );
        cxcoutPHF << "Successfully wrote channel to file" << std::endl;
      }

      channel_workspaces.emplace_back(std::move(ws_single));
    } // End loop over channels

    /***
   Second: Make the combined model:
   If you want output histograms in root format, create and pass it to the combine routine.
   "combine" : will do the individual cross-section measurements plus combination
    ***/

    // Use HistFactory to combine the individual channel workspaces
    std::unique_ptr<RooWorkspace> ws{factory.MakeCombinedModel(channel_names, channel_workspaces)};

    // Configure that workspace
    HistoToWorkspaceFactoryFast::ConfigureWorkspaceForMeasurement("simPdf", ws.get(), measurement);

    {
      std::string CombinedFileName = measurement.GetOutputFilePrefix() + "_combined_"
        + rowTitle + "_model.root";
      cxcoutPHF << "Writing combined workspace to file: " << CombinedFileName << std::endl;
      std::unique_ptr<TFile> combFile{TFile::Open( CombinedFileName.c_str(), "RECREATE" )};
      if( combFile == nullptr ) {
        cxcoutEHF << "Error: Failed to open file " << CombinedFileName << std::endl;
        throw hf_exc();
      }
      combFile->WriteTObject(ws.get());
      cxcoutPHF << "Writing combined measurement to file: " << CombinedFileName << std::endl;
      measurement.writeToFile( combFile.get() );
    }

  msgSvc.getStream(1).addTopic(RooFit::ObjectHandling);

  return RooFit::makeOwningPtr(std::move(ws));
}
