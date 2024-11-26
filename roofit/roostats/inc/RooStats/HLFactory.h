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

#ifndef RooStats_HLFactory
#define RooStats_HLFactory

#include "RooAbsPdf.h"
#include "RooCategory.h"
#include "RooDataSet.h"
#include "RooWorkspace.h"

#include <ROOT/RConfig.hxx> // for R__DEPRECATED

 class TString;

namespace RooStats {

  class HLFactory : public TNamed {

  public:

    /// Constructor
    HLFactory(const char *name,
              const char *fileName=nullptr,
              bool isVerbose = false);

    /// Constructor with external RooWorkspace
    HLFactory(const char* name,
              RooWorkspace* externalWs,
              bool isVerbose = false);

    /// Default Constructor
    HLFactory();

    /// Default Destructor
    ~HLFactory() override;

    /// Add channel for the combination
    int AddChannel(const char* label,
                   const char* SigBkgPdfName,
                   const char* BkgPdfName=nullptr,
                   const char* datasetName=nullptr);

    /// Dump the Workspace content as configuration file
    /* It needs some workspace object list or something..*/
     void DumpCfg(const char* /*cardname*/ ){ /* t.b.i. */ }; // Dump the factory content as configuration file

    /// Get the combined signal plus background pdf
    RooAbsPdf* GetTotSigBkgPdf(); // Get the Signal and Background combined model

    /// Get the combined background pdf
    RooAbsPdf* GetTotBkgPdf(); // Get the Background combined model

    /// Get the combined dataset
    RooDataSet* GetTotDataSet(); // Get the combined dataset

    /// Get the combined dataset
    RooCategory* GetTotCategory(); // Get the category

    /// Get the RooWorkspace containing the models and variables
    RooWorkspace* GetWs(){return fWs;}; // Expose the internal Workspace

    /// Process a configuration file
    int ProcessCard(const char* filename);

  private:

    /// Create the category for the combinations
    void fCreateCategory();

    /// Check the length of the lists
    bool fNamesListsConsistent();

    /// Read the actual cfg file
    int fReadFile(const char*fileName, bool is_included = false);

    /// Parse a single line an puts the content in the RooWorkSpace
    int fParseLine(TString& line);

    RooCategory *fComboCat = nullptr;     ///< The category of the combination
    RooAbsPdf *fComboBkgPdf = nullptr;    ///< The background model combination
    RooAbsPdf *fComboSigBkgPdf = nullptr; ///< The signal plus background model combination
    RooDataSet *fComboDataset = nullptr;  ///< The datasets combination
    bool fCombinationDone = false;        ///< Flag to keep trace of the status of the combination
    TList fSigBkgPdfNames;                ///< List of channels names to combine for the signal plus background pdfs
    TList fBkgPdfNames;                   ///< List of channels names to combine for the background pdfs
    TList fDatasetsNames;                 ///< List of channels names to combine for the datasets
    TList fLabelsNames;                   ///< List of channels names to combine for the datasets
    bool fVerbose = false;                ///< The verbosity flag
    int fInclusionLevel = 0;              ///< Keep trace of the inclusion deepness
    RooWorkspace *fWs = nullptr;          ///< The RooWorkspace containing the models and variables
    bool fOwnWs = false;                  ///< Owns workspace

  ClassDefOverride(HLFactory,1)  // The high Level Model Factory to create models from datacards

#ifdef ROOFIT_BUILDS_ITSELF // to avoid the deprecation warnings when building RooFit itself
  };
#else
  } R__DEPRECATED(6,38, "Outdated higher-level interface around RooWorkspace. Use RooWorkspace directly.");
#endif
}

#endif
