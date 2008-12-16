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

#ifndef ROOSTATS_HybridCalculator
#define ROOSTATS_HybridCalculator

#ifndef ROOSTATS_HypoTestCalculator
#include "RooStats/HypoTestCalculator.h"
#endif

#include <vector>


#ifndef ROOSTATS_HypoTestResult
#include "RooStats/HybridResult.h"
#endif

class TH1; 

namespace RooStats {

   class HybridCalculator : public HypoTestCalculator , public TNamed {

   public:


      /// Constructor with only name and title 
      HybridCalculator(const char *name = 0,
                       const char *title = 0); 
      
      /// Constructor for HybridCalculator
      HybridCalculator(const char *name,
                       const char *title,
                       RooAbsPdf& sb_model,
                       RooAbsPdf& b_model,
                       RooArgList& observables,
                       RooArgSet& nuisance_parameters,
                       RooAbsPdf& prior_pdf);

      /// Constructor for HybridCalculator using  a data set and pdf instances
      HybridCalculator(RooAbsData& data, 
                       RooAbsPdf& sb_model,
                       RooAbsPdf& b_model,
                       RooArgSet* nuisance_parameters,
                       RooAbsPdf* prior_pdf);

      /// Constructor for HybridCalculator using name, title, a data set and pdf instances
      HybridCalculator(const char *name,
                       const char *title,
                       RooAbsData& data, 
                       RooAbsPdf& sb_model,
                       RooAbsPdf& b_model,
                       RooArgSet* nuisance_parameters,
                       RooAbsPdf* prior_pdf);


   private: // not yet available 
 
      /// Constructor for HybridCalculator using name, title, a workspace and pdf names
      HybridCalculator(RooWorkspace & wks, 
                       const char* data, 
                       const char* sb_model,
                       const char* b_model,
                       RooArgSet* nuisance_parameters,
                       const char* prior_pdf);

      /// Constructor for HybridCalculator using name, title, a workspace and pdf names
      HybridCalculator(const char *name,
                       const char *title,
                       RooWorkspace & wks, 
                       const char* data, 
                       const char* sb_model,
                       const char* b_model,
                       RooArgSet* nuisance_parameters,
                       const char* prior_pdf);

   public: 

      /// Destructor of HybridCalculator
      virtual ~HybridCalculator();

      /// inherited methods from HypoTestCalculanterface
      virtual HybridResult* GetHypoTest() const;

      // inherited setter methods from HypoTestCalculator

   private: 
      // set a workspace that owns all the necessary components for the analysis
      virtual void SetWorkspace(RooWorkspace& ws);
      // set the PDF for the null hypothesis (only B)
      virtual void SetNullPdf(const char* name) { fBModelName = name; }
      // set the PDF for the alternate hypothesis  (S+B)
      virtual void SetAlternatePdf(const char* name ) { fSbModelName = name;} 
      // set a common PDF for both the null and alternate hypotheses
      virtual void SetCommonPdf(const char* name) {fSbModelName = name; }

   public: 

      // Set a common PDF for both the null and alternate
      virtual void SetCommonPdf(RooAbsPdf & pdf) { fSbModel = &pdf; }
      // Set the PDF for the null (only B)
      virtual void SetNullPdf(RooAbsPdf& pdf) { fBModel = &pdf; }
      // Set the PDF for the alternate hypothesis ( i.e. S+B)
      virtual void SetAlternatePdf(RooAbsPdf& pdf) { fSbModel = &pdf;  }

      // specify the name of the dataset in the workspace to be used
      virtual void SetData(const char* name) { fDataName = name; } 
      // Set the DataSet, add to the the workspace if not already there
      virtual void SetData(RooAbsData& data) { fData = &data; }

      // set parameter values for the null if using a common PDF
      virtual void SetNullParameters(RooArgSet& ) { } // not needed
      // set parameter values for the alternate if using a common PDF
      virtual void SetAlternateParameters(RooArgSet&) {}  // not needed

      // additional methods specific for HybridCalculator
      // set a  prior pdf for the nuisance parameters 
      void SetNuisancePdf(RooAbsPdf & prior_pdf) {          
         fPriorPdf = &prior_pdf; 
         fUsePriorPdf = true; // if set by default turn it on
      } 

      // set name of a  prior pdf for the nuisance parameters in the previously given workspace
      void SetNuisancePdf(const char * name) { 
         fPriorPdfName = name; 
         fUsePriorPdf = true; // if set by default turn it on
      } 
      
      // set the nuisance parameters to be marginalized
      void SetNuisanceParameters(RooArgSet & params) { fParameters = &params; }

      // set number of toy MC 
      void SetNumberOfToys(unsigned int ntoys) { fNToys = ntoys; }

      // control use of the pdf for the nuisance parameter and marginalize them
      void UseNuisance(bool on = true) { fUsePriorPdf = on; }
      
      void SetTestStatistics(int index);

      HybridResult* Calculate(TH1& data, unsigned int nToys, bool usePriors) const;
      HybridResult* Calculate(RooTreeData& data, unsigned int nToys, bool usePriors) const;
      HybridResult* Calculate(unsigned int nToys, bool usePriors) const;
      void PrintMore(const char* options) const;


   private:
      void RunToys(std::vector<double>& bVals, std::vector<double>& sbVals, unsigned int nToys, bool usePriors) const;

      // check input parameters before performing the calculation
      bool DoCheckInputs() const; 
      // initialize all the data and pdf by using a workspace as input 
      bool DoInitializeFromWS();  

      

      unsigned int fTestStatisticsIdx; // Index of the test statistics to use
      unsigned int fNToys;            // number of Toys MC
      bool  fUsePriorPdf;               // use a prior for nuisance parameters  

      RooAbsPdf* fSbModel; // The pdf of the signal+background model
      RooAbsPdf* fBModel; // The pdf of the background model
      mutable RooArgList* fObservables; // Collection of the observables of the model
      RooArgSet* fParameters; // Collection of the nuisance parameters in the model
      RooAbsPdf* fPriorPdf; // Prior PDF of the nuisance parameters
      RooAbsData * fData;     // pointer to the data sets 
      //bool fOwnsWorkspace;    // flag indicating if calculator manages the workspace 
      RooWorkspace * fWS;     // a workspace that owns all the components to be used by the calculator
      TString fSbModelName;   // name of pdf of the signal+background model
      TString fBModelName;   // name of pdf of the background model
      TString fPriorPdfName;   // name of pdf of the background model
      TString fDataName;      // name of the dataset in the workspace

   protected:
      ClassDef(HybridCalculator,1)  // Hypothesis test calculator using a Bayesian-frequentist hybrid method
   };
}

#endif
