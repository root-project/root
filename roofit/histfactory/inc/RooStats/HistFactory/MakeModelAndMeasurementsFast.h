#ifndef HISTFACTORY_MAKEMODELANDMEASUREMENTSFAST_H
#define HISTFACTORY_MAKEMODELANDMEASUREMENTSFAST_H

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"

#include "RooWorkspace.h"
#include "RooPlot.h"

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

  }
}


#endif
