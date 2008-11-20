// @(#)root/roostats:$Id$

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke       *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
/// BEGIN_HTML
/// HybridResult class: this class is a fresh rewrite in RooStats of
/// 	RooStatsCms/LimitResults developped by D. Piparo and G. Schott
/// END_HTML
///////////////////////////////////////////////////////////////////////////

#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooGlobalFunc.h" // for RooFit::Extended()
#include "RooNLLVar.h"
#include "RooRealVar.h"
#include "RooTreeData.h"

#include "RooStats/HybridResult.h"
#include "RooStats/HybridPlot.h"


/// ClassImp for building the THtml documentation of the class 
ClassImp(RooStats::HybridResult)

using namespace RooStats;

///////////////////////////////////////////////////////////////////////////

HybridResult::HybridResult( const char *name, const char *title,
                            std::vector<float>& testStat_sb_vals,
                            std::vector<float>& testStat_b_vals,
                            float testStat_data_val ) :
   /*HypoTestCalculator(name,title),*/ /// TO DO
   fName(name),
   fTitle(title),
   fTestStat_data(testStat_data_val)
{
   /// HybridResult constructor:
   int vector_size_sb = testStat_sb_vals.size();
   assert(vector_size_sb>0);

   int vector_size_b = testStat_b_vals.size();
   assert(vector_size_b>0);

   fTestStat_sb.reserve(vector_size_sb);
   fTestStat_b.reserve(vector_size_b);

   for (int i=0;i<vector_size_sb;++i)
      fTestStat_sb.push_back(testStat_sb_vals[i]);

   for (int i=0;i<vector_size_b;++i)
      fTestStat_b.push_back(testStat_b_vals[i]);

//  _testStat_data = -999;
}

///////////////////////////////////////////////////////////////////////////

HybridResult::~HybridResult()
{
   /// HybridResult destructor
   fTestStat_sb.clear();
   fTestStat_b.clear();
}

///////////////////////////////////////////////////////////////////////////

double HybridResult::CLb()
{
   /// return CL_b

   int nToys = fTestStat_b.size();
   if (nToys==0) {
      std::cout << "Error: no toy data present. Returning -1.\n";
      // TO DO: assert
      return -1;
   }

   double larger_than_measured=0;
   for (int iToy=0;iToy<nToys;++iToy)
      if ( fTestStat_b[iToy] > fTestStat_data ) ++larger_than_measured;

   if (larger_than_measured==0) std::cout << "Warning: CLb = 0 ... maybe more toys are needed!\n";
   return larger_than_measured/nToys;
}

///////////////////////////////////////////////////////////////////////////

double HybridResult::CLsplusb()
{
   /// return CL_s+b
   int nToys = fTestStat_b.size();
   if (nToys==0) {
      std::cout << "Error: no toy data present. Returning -1.\n";
      // TO DO: assert!
      return -1;
   }

   double larger_than_measured=0;
   for (int iToy=0;iToy<nToys;++iToy)
      if ( fTestStat_sb[iToy] > fTestStat_data ) ++larger_than_measured;

   if (larger_than_measured==0) std::cout << "Warning: CLsb = 0 ... maybe more toys are needed!\n";
   return larger_than_measured/nToys;
}

///////////////////////////////////////////////////////////////////////////

double HybridResult::CLs()
{
   /// returns CL_s = CL_s+b / CL_b
   double thisCLb = CLb();
   if (thisCLb==0) {
      std::cout << "Error: Cannot compute CLs because CLb = 0. Returning CLs = -1\n";
      // TO DO: assert!
      return -1;
   }
   double thisCLsb = CLsplusb();

   return thisCLsb/thisCLb;
}

///////////////////////////////////////////////////////////////////////////

void HybridResult::Add(HybridResult* other)
{
/// TO DO: to complete
   other->Print("");

//   int other_size_sb = _testStat_sb.size();
//   for (int i=0;i<other_size_sb;++i)
//     _testStat_sb.push_back(other->getTestStat_sb()[i]);
// 
//   int other_size_b = _testStat_b.size();
//   for (int i=0;i<other_size_b;++i)
//     _testStat_b.push_back(other->getTestStat_b()[i]);
// 
//   // if no data is present use the other's HybridResult's data
//   if (_testStat_data==-999)
//     _testStat_data = other->getTestStat_data();

   return;
}

///////////////////////////////////////////////////////////////////////////

HybridPlot* HybridResult::GetPlot(const char* name,const char* title, int n_bins)
{
   // TO DO: migrate to roostats

   // default plot name
   TString plot_name;
   if ( TString(name)=="" ) {
      plot_name += fName; //GetName();
      plot_name += "_plot";
   } else plot_name = name;

   // default plot title
   TString plot_title;
   if ( TString(title)=="" ) {
      plot_title += fTitle; //GetTitle();
      plot_title += "_plot (";
      plot_title += fTestStat_b.size();
      plot_title += " toys)";
   } else plot_title = title;

   HybridPlot* plot = new HybridPlot( plot_name.Data(),
                                      plot_title.Data(),
                                      fTestStat_sb,
                                      fTestStat_b,
                                      fTestStat_data,
                                      n_bins,
                                      true );
   return plot;
}

///////////////////////////////////////////////////////////////////////////

void HybridResult::Print(const char* options)
{
   /// Print out some information about the results

   std::cout << options << std::endl;


   std::cout << "\nResults " << fName /*GetName()*/ << ":\n"
             << " - Number of S+B toys: " << fTestStat_b.size() << std::endl
             << " - Number of B toys: " << fTestStat_sb.size() << std::endl
             << " - test statistics evaluated on data: " << fTestStat_data << std::endl
             << " - CL_b " << CLb() << std::endl
             << " - CL_s+b " << CLsplusb() << std::endl
             << " - CL_s " << CLs() << std::endl;

   return;
}










