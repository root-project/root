// @(#)root/roostats:$Id$

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Danilo Piparo, Gregory Schott                                       *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HybridCalculatorOriginal
#define ROOSTATS_HybridCalculatorOriginal

#ifndef ROOSTATS_HypoTestCalculator
#include "RooStats/HypoTestCalculator.h"
#endif

#include <vector>


#ifndef ROOSTATS_HybridResult
#include "RooStats/HybridResult.h"
#endif

#ifndef ROOSTATS_ModelConfig
#include "RooStats/ModelConfig.h"
#endif

class TH1; 

namespace RooStats {

   class HybridResult; 

   class HybridCalculatorOriginal : public HypoTestCalculator , public TNamed {

   public:


      /// Dummy Constructor with only name 
      explicit HybridCalculatorOriginal(const char *name = 0);
      
      /// Constructor for HybridCalculator from pdf instances but without a data-set
      HybridCalculatorOriginal(RooAbsPdf& sb_model,
                       RooAbsPdf& b_model,
                       RooArgList& observables,
                       const RooArgSet* nuisance_parameters = 0,
                       RooAbsPdf* prior_pdf = 0,
                       bool GenerateBinned = false, int testStatistics = 1, int ntoys = 1000 );

      /// Constructor for HybridCalculator using  a data set and pdf instances
      HybridCalculatorOriginal(RooAbsData& data,
                       RooAbsPdf& sb_model,
                       RooAbsPdf& b_model,
                       const RooArgSet* nuisance_parameters = 0,
                       RooAbsPdf* prior_pdf = 0,
                       bool GenerateBinned = false, int testStatistics = 1, int ntoys = 1000 );


      /// Constructor passing a ModelConfig for the SBmodel and a ModelConfig for the B Model
      HybridCalculatorOriginal(RooAbsData& data,
                       const ModelConfig& sb_model, 
                       const ModelConfig& b_model,
                       bool GenerateBinned = false, int testStatistics = 1, int ntoys = 1000 );


   public: 

      /// Destructor of HybridCalculator
      virtual ~HybridCalculatorOriginal();

      /// inherited methods from HypoTestCalculator interface
      virtual HybridResult* GetHypoTest() const;

      // inherited setter methods from HypoTestCalculator


      // set the model for the null hypothesis (only B)
      virtual void SetNullModel(const ModelConfig & );
      // set the model for the alternate hypothesis  (S+B)
      virtual void SetAlternateModel(const ModelConfig & );


      // Set a common PDF for both the null and alternate
      virtual void SetCommonPdf(RooAbsPdf & pdf) { fSbModel = &pdf; }
      // Set the PDF for the null (only B)
      virtual void SetNullPdf(RooAbsPdf& pdf) { fBModel = &pdf; }
      // Set the PDF for the alternate hypothesis ( i.e. S+B)
      virtual void SetAlternatePdf(RooAbsPdf& pdf) { fSbModel = &pdf;  }

      // Set the DataSet
      virtual void SetData(RooAbsData& data) { fData = &data; }

      // set parameter values for the null if using a common PDF
      virtual void SetNullParameters(const RooArgSet& ) { } // not needed
      // set parameter values for the alternate if using a common PDF
      virtual void SetAlternateParameters(const RooArgSet&) {}  // not needed

      // additional methods specific for HybridCalculator
      // set a  prior pdf for the nuisance parameters 
      void SetNuisancePdf(RooAbsPdf & prior_pdf) {          
         fPriorPdf = &prior_pdf; 
         fUsePriorPdf = true; // if set by default turn it on
      } 
      
      // set the nuisance parameters to be marginalized
      void SetNuisanceParameters(const RooArgSet & params) { fNuisanceParameters = &params; }

      // set number of toy MC (Default is 1000)
      void SetNumberOfToys(unsigned int ntoys) { fNToys = ntoys; }

      // return number of toys used
      unsigned int GetNumberOfToys() const { return fNToys; }

      // control use of the pdf for the nuisance parameter and marginalize them
      void UseNuisance(bool on = true) { fUsePriorPdf = on; }

      // control to use bin data generation 
      void SetGenerateBinned(bool on = true) { fGenerateBinned = on; }

      /// set the desired test statistics:
      /// index=1 : 2 * log( L_sb / L_b )  (DEFAULT)
      /// index=2 : number of generated events
      /// index=3 : profiled likelihood ratio
      /// if the index is different to any of those values, the default is used
      void SetTestStatistic(int index);

      HybridResult* Calculate(TH1& data, unsigned int nToys, bool usePriors) const;
      HybridResult* Calculate(RooAbsData& data, unsigned int nToys, bool usePriors) const;
      HybridResult* Calculate(unsigned int nToys, bool usePriors) const;
      void PrintMore(const char* options) const;

      void PatchSetExtended(bool on = true) { fTmpDoExtended = on; std::cout << "extended patch set to " << on << endl; } // patch to test with RooPoisson (or other non-extended models)

   private:

      void RunToys(std::vector<double>& bVals, std::vector<double>& sbVals, unsigned int nToys, bool usePriors) const;

      // check input parameters before performing the calculation
      bool DoCheckInputs() const; 

      unsigned int fTestStatisticsIdx; // Index of the test statistics to use
      unsigned int fNToys;            // number of Toys MC
      RooAbsPdf* fSbModel; // The pdf of the signal+background model
      RooAbsPdf* fBModel; // The pdf of the background model
      mutable RooArgList* fObservables; // Collection of the observables of the model
      const RooArgSet* fNuisanceParameters;   // Collection of the nuisance parameters in the model
      RooAbsPdf* fPriorPdf;   // Prior PDF of the nuisance parameters
      RooAbsData * fData;     // pointer to the data sets 
      bool fGenerateBinned;   //Flag to control binned generation
      bool  fUsePriorPdf;               // use a prior for nuisance parameters  
      bool fTmpDoExtended;

//       TString fSbModelName;   // name of pdf of the signal+background model
//       TString fBModelName;   // name of pdf of the background model
//       TString fPriorPdfName;   // name of pdf of the background model
//       TString fDataName;      // name of the dataset in the workspace

   protected:
      ClassDef(HybridCalculatorOriginal,1)  // Hypothesis test calculator using a Bayesian-frequentist hybrid method
   };

}

#endif
