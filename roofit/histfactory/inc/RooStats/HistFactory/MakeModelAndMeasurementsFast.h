
#ifndef HISTFACTORY_MAKEMODELANDMEASUREMENTSFAST_H
#define HISTFACTORY_MAKEMODELANDMEASUREMENTSFAST_H

#include <string>
#include <vector>

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/EstimateSummary.h"

#include "RooWorkspace.h"
#include "RooPlot.h"
#include "TFile.h"



namespace RooStats{
  namespace HistFactory{

    //void fastDriver(std::string input);

    RooWorkspace* MakeModelAndMeasurementFast( RooStats::HistFactory::Measurement& measurement );
    //RooWorkspace* MakeModelFast( RooStats::HistFactory::Measurement& measurement );

    std::vector<RooStats::HistFactory::EstimateSummary> GetChannelEstimateSummaries(RooStats::HistFactory::Measurement& measurement, RooStats::HistFactory::Channel& channel);
    // void ConfigureWorkspaceForMeasurement( const std::string&, RooWorkspace*, RooStats::HistFactory::Measurement&);

    void FormatFrameForLikelihood(RooPlot* frame, std::string xTitle=std::string("#sigma / #sigma_{SM}"), std::string yTitle=std::string("-log likelihood"));
    void FitModel(RooWorkspace *, std::string data_name="obsData");
    void FitModelAndPlot(const std::string& measurementName, const std::string& fileNamePrefix, RooWorkspace *, std::string, std::string, TFile*, FILE*);
  }
}


#endif
