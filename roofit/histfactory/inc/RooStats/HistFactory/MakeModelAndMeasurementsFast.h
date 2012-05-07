
#ifndef MAKEMODELANDMEASUREMENTSFAST_H
#define MAKEMODELANDMEASUREMENTSFAST_H


namespace RooStats{
  namespace HistFactory{


    RooWorkspace* MakeModelAndMeasurementFast( RooStats::HistFactory::Measurement& measurement );
    //RooWorkspace* MakeModelFast( RooStats::HistFactory::Measurement& measurement );

    std::vector<EstimateSummary> GetChannelEstimateSummaries(RooStats::HistFactory::Measurement& measurement, RooStats::HistFactory::Channel& channel);
    // void ConfigureWorkspaceForMeasurement( const std::string&, RooWorkspace*, RooStats::HistFactory::Measurement&);

    void FormatFrameForLikelihood(RooPlot* frame, string XTitle=string("#sigma / #sigma_{SM}"), string YTitle=string("-log likelihood"));
    void FitModel(RooWorkspace *, string data_name="obsData");
    void FitModelAndPlot(const std::string& MeasurementName, const std::string& FileNamePrefix, RooWorkspace *, string, string, TFile*, FILE*);
  }
}


#endif
