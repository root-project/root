// @(#)root/roostats:$Id$
// Author: Kyle Cranmer and Sven Kreiss  July 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ProofConfig
#define ROOSTATS_ProofConfig

//_________________________________________________
/*
BEGIN_HTML
<p>
Holds configuration options for proof and proof-lite.

This class will be expanded in the future to hold more specific configuration
options for the tools in RooStats.
</p>


<p>
Access to TProof::Mgr for configuration is still possible as usual
(e.g. to set Root Version to be used on workers). You can do:</p>
<ul>
  <li>TProof::Mgr("my.server.url")->ShowROOTVersions()</li>
  <li>TProof::Mgr("my.server.url")->SetROOTVersion("v5-27-06_dbg")</li>
</ul>

<p>
See doc: http://root.cern.ch/drupal/content/changing-default-root-version
</p>
END_HTML
*/
//

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "RooWorkspace.h"
#include "RooStudyManager.h"

namespace RooStats {

class ProofConfig {

   public:
      ProofConfig(RooWorkspace &w, Int_t nExperiments = 8, const char *host = "", Bool_t showGui = kFALSE) :
         fWorkspace(w),
         fNExperiments(nExperiments),
         fHost(host),
         fShowGui(showGui)
      {
      }

      virtual ~ProofConfig() {
         ProofConfig::CloseProof();
      }

      /// close all proof connections
      static void CloseProof(Option_t *option = "s") { RooStudyManager::closeProof(option); }

      // returns fWorkspace
      RooWorkspace& GetWorkspace(void) { return fWorkspace; }
      // returns fHost
      const char* GetHost(void) { return fHost; }
      // return fNExperiments
      Int_t GetNExperiments(void) { return fNExperiments; }
      // return fShowGui
      Bool_t GetShowGui(void) { return fShowGui; }

   protected:
      RooWorkspace& fWorkspace;   // workspace that is to be used with the RooStudyManager
      Int_t fNExperiments;        // number of experiments. This is sometimes called "events" in proof; "experiments" in RooStudyManager.
      const char* fHost;          // Proof hostname. Use empty string (ie "") for proof-lite. Can also handle options like "workers=2" to run on two nodes.
      Bool_t fShowGui;            // Whether to show the Proof Progress window.

   protected:
   ClassDef(ProofConfig,1) // Configuration options for proof.
};
}


#endif
