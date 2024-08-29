#ifndef HISTFACTORY_MAKEMODELANDMEASUREMENTSFAST_H
#define HISTFACTORY_MAKEMODELANDMEASUREMENTSFAST_H

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"

#include "RooWorkspace.h"
#include "RooPlot.h"

#include <ROOT/RConfig.hxx> // for the R__DEPRECATED macro

#include <iostream>
#include <string>
#include <vector>

class TFile;

namespace RooStats{
  namespace HistFactory{

      RooFit::OwningPtr<RooWorkspace> MakeModelAndMeasurementFast(
            RooStats::HistFactory::Measurement& measurement,
            HistoToWorkspaceFactoryFast::Configuration const& cfg={}
    );

    void FormatFrameForLikelihood(RooPlot* frame, std::string xTitle=std::string("#sigma / #sigma_{SM}"), std::string yTitle=std::string("-log likelihood"))
#ifndef ROOFIT_BUILDS_ITSELF
        R__DEPRECATED(6,36, "Please write your own plotting code inspired by the hf001 tutorial.")
#endif
        ;
    void FitModel(RooWorkspace *, std::string data_name="obsData")
        R__DEPRECATED(6,36, "Please write your own plotting code inspired by the hf001 tutorial.");
    void FitModelAndPlot(const std::string& measurementName, const std::string& fileNamePrefix, RooWorkspace &, std::string, std::string, TFile&, std::ostream&)
#ifndef ROOFIT_BUILDS_ITSELF
        R__DEPRECATED(6,36, "Please write your own plotting code inspired by the hf001 tutorial.")
#endif
        ;
  }
}


#endif
