// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOSTATS_ESTIMATESUMMARY_h
#define ROOSTATS_ESTIMATESUMMARY_h

#include "TH1F.h"
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "TObject.h"


namespace RooStats{
namespace HistFactory{

struct  EstimateSummary : public TObject {
      
   
   struct NormFactor{
      std::string name;
      double val, high, low;
      bool constant;
   };
   
   typedef std::vector<std::string> vecstring;
   typedef std::vector<TH1F*> vechist;
   typedef std::pair<double, double> pairdouble;
   typedef std::map<std::string, std::pair<double, double> > mappair;
      
   // simple structure to hold necessary information about each channel
   EstimateSummary();
   virtual ~EstimateSummary();
   void Print(const char *opt = 0) const ;
   void AddSyst( const std::string & sname, TH1F* low, TH1F* high);
   bool operator==(const EstimateSummary &other) const ;
   bool CompareHisto( const TH1 * one, const TH1 * two) const ;
   
   
   //data members .
   std::string name; 
   std::string channel; 
   std::string normName;
   TH1F* nominal;  // x pb per jet bin.  all histograms need index of binning to be consistent
   std::vector<std::string> systSourceForHist;
   std::vector<TH1F*> lowHists; // x pb per jet bin for - variations over list of systematics
   std::vector<TH1F*> highHists; // x pb per jet bin for + variations over list of systematics
   std::map<std::string, std::pair<double, double> > overallSyst; // "acceptance"->(0.8,1.2)
   std::pair<double, double> dummyForRoot;
   std::vector<NormFactor> normFactor;
   
   ClassDef(RooStats::HistFactory::EstimateSummary,1)
};

}
}

#endif
