// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HELPER
#define ROOSTATS_HELPER

#include <string>
#include <vector>
#include <map>

#include "TFile.h"

#include "RooStats/HistFactory/EstimateSummary.h"
using namespace std; 

namespace RooStats{
namespace HistFactory{
  vector<EstimateSummary>*  loadSavedInputs(TFile* outFile, string channel );
  void saveInputs(TFile* outFile, string channel, vector<EstimateSummary> summaries);
  TH1 * GetHisto( TFile * inFile, const string name );
  TH1 * GetHisto( const string file, const string path, const string obj );
  bool AddSummaries( vector<EstimateSummary> & summary, vector<vector<EstimateSummary> > &master);
  vector<pair<string, string> > get_comb(vector<string> names);
  void AddSubStrings( vector<string> & vs, string s);
}
}
#endif
