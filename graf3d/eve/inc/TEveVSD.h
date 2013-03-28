// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveVSD
#define ROOT_TEveVSD

#include "TEveUtil.h"
#include "TEveVSDStructs.h"
#include "TTree.h"

class TEveVSD : public TObject
{
   TEveVSD(const TEveVSD&);            // Not implemented
   TEveVSD& operator=(const TEveVSD&); // Not implemented

protected:
   TFile             *fFile;        //!
   TDirectory        *fDirectory;   //!

   Int_t              fBuffSize;    //!
   Int_t              fVerbose;     //!

public:
   TTree*             fTreeK;       //! Kinematics.
   TTree*             fTreeH;       //! Hits.
   TTree*             fTreeC;       //! Clusters.
   TTree*             fTreeR;       //! Reconstructed tracks.
   TTree*             fTreeKK;      //! Kinks.
   TTree*             fTreeV0;      //! VO's.
   TTree*             fTreeCC;      //! Cascades.
   TTree*             fTreeGI;      //! Sim-Rec cross references.

   TEveMCTrack        fK,  *fpK;    //!
   TEveHit            fH,  *fpH;    //!
   TEveCluster        fC,  *fpC;    //!
   TEveRecTrack       fR,  *fpR;    //!
   TEveRecKink        fKK, *fpKK;   //!
   TEveRecV0          fV0, *fpV0;   //!
   TEveRecCascade     fCC, *fpCC;   //!
   TEveMCRecCrossRef  fGI, *fpGI;   //!

public:
   TEveVSD(const char* name="TEveVSD", const char* title="");
   virtual ~TEveVSD();

   virtual void SetDirectory(TDirectory* dir);

   virtual void CreateTrees();
   virtual void DeleteTrees();

   virtual void CreateBranches();
   virtual void WriteTrees();

   virtual void LoadTrees();
   virtual void SetBranchAddresses();

   static void DisableTObjectStreamersForVSDStruct();

   ClassDef(TEveVSD, 1); // Visualization Summary Data - a collection of trees holding standard event data in experiment independent format.
};

#endif
