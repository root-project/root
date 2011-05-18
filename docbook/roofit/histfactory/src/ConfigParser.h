// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cstdlib>
#include <string>
#include <TXMLNode.h>

#include "TList.h"
#include "TFile.h"
#include "TXMLAttr.h"

#include "RooStats/HistFactory/EstimateSummary.h"
using namespace std; 

// KC: Should make this a class and have it do some of what is done in MakeModelAndMeasurements

namespace RooStats{
   namespace HistFactory {

     typedef pair<double,double> UncertPair;
     void ReadXmlConfig( string, vector<RooStats::HistFactory::EstimateSummary>& , Double_t );
     void AddSystematic( RooStats::HistFactory::EstimateSummary &, TXMLNode*, string, string,string);
   }
}
