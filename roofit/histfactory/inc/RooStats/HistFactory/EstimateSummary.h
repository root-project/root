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

#include "TH1.h"
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "TObject.h"


namespace RooStats{
namespace HistFactory{

struct  EstimateSummary : public TObject {

  enum ConstraintType{ Gaussian, Poisson };            
   
   struct NormFactor{
      std::string name;
      double val, high, low;
      bool constant;
   };


   class ShapeSys{
   public:
     ShapeSys() : name(), hist(nullptr), constraint{} {;}
     std::string name;
     const TH1* hist;
     ConstraintType constraint;
   };
      

   typedef std::vector<std::string> vecstring;
   typedef std::vector<TH1*> vechist;
   typedef std::pair<double, double> pairdouble;
   typedef std::map<std::string, std::pair<double, double> > mappair;
      
   // simple structure to hold necessary information about each channel
   EstimateSummary();
   virtual ~EstimateSummary();
   void Print(const char *opt = 0) const ;
   void AddSyst( const std::string & sname, TH1* low, TH1* high);
   bool operator==(const EstimateSummary &other) const ;
   bool CompareHisto( const TH1 * one, const TH1 * two) const ;
   
   
   //data members .
   std::string name; 
   std::string channel; 
   std::string normName;
   TH1* nominal;  // x pb per jet bin.  all histograms need index of binning to be consistent
   std::vector<std::string> systSourceForHist;
   std::vector<TH1*> lowHists; // x pb per jet bin for - variations over list of systematics
   std::vector<TH1*> highHists; // x pb per jet bin for + variations over list of systematics
   std::map<std::string, std::pair<double, double> > overallSyst; // "acceptance"->(0.8,1.2)
   std::pair<double, double> dummyForRoot;
   std::vector<NormFactor> normFactor;



  bool IncludeStatError; // Flag to implement Statistical errors for this sample
  ConstraintType StatConstraintType;  // The type of constraint binwise stat errors
  Double_t RelErrorThreshold; // The minimum relative uncertainty for a bin to use stat errors
  TH1* relStatError; // An (optional) externally provided shape for this error

  //  bool doShapeFactor; // A flag to include a ShapeFactor ParamatarizedHistogram
  std::string shapeFactorName; //
  std::vector<ShapeSys> shapeSysts; //

   ClassDef(RooStats::HistFactory::EstimateSummary,1)
};

}
}

#endif
