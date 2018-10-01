// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REveVSD_hxx
#define ROOT_REveVSD_hxx

#include "ROOT/REveUtil.hxx"
#include "ROOT/REveVSDStructs.hxx"
#include "TTree.h"

namespace ROOT {
namespace Experimental {

class REveVSD : public TObject {
   REveVSD(const REveVSD &);            // Not implemented
   REveVSD &operator=(const REveVSD &); // Not implemented

protected:
   TFile *fFile;           //!
   TDirectory *fDirectory; //!

   Int_t fBuffSize; //!
   Int_t fVerbose;  //!

public:
   TTree *fTreeK;  //! Kinematics.
   TTree *fTreeH;  //! Hits.
   TTree *fTreeC;  //! Clusters.
   TTree *fTreeR;  //! Reconstructed tracks.
   TTree *fTreeKK; //! Kinks.
   TTree *fTreeV0; //! VO's.
   TTree *fTreeCC; //! Cascades.
   TTree *fTreeGI; //! Sim-Rec cross references.

   REveMCTrack fK, *fpK;         //!
   REveHit fH, *fpH;             //!
   REveCluster fC, *fpC;         //!
   REveRecTrack fR, *fpR;        //!
   REveRecKink fKK, *fpKK;       //!
   REveRecV0 fV0, *fpV0;         //!
   REveRecCascade fCC, *fpCC;    //!
   REveMCRecCrossRef fGI, *fpGI; //!

public:
   REveVSD(const char *name = "REveVSD", const char *title = "");
   virtual ~REveVSD();

   virtual void SetDirectory(TDirectory *dir);

   virtual void CreateTrees();
   virtual void DeleteTrees();

   virtual void CreateBranches();
   virtual void WriteTrees();

   virtual void LoadTrees();
   virtual void SetBranchAddresses();

   static void DisableTObjectStreamersForVSDStruct();

   ClassDef(REveVSD, 1); // Visualization Summary Data - a collection of trees holding standard event data in experiment independent format.
};

} // namespace Experimental
} // namespace ROOT

#endif
