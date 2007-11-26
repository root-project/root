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

#include <TEveUtil.h>
#include <TEveVSDStructs.h>
#include <TTree.h>

class TEveVSD : public TObject
{
   TEveVSD(const TEveVSD&);            // Not implemented
   TEveVSD& operator=(const TEveVSD&); // Not implemented

protected:
   Int_t        fBuffSize;

   TFile*       mFile;        //!
   TDirectory*  mDirectory;   //!

public:
   TTree*       mTreeK;       //! X{g}
   // TTree*    mTreeTR;      //! X{g}
   TTree*       mTreeH;       //! X{g}
   TTree*       mTreeC;       //! X{g}
   TTree*       mTreeR;       //! X{g}
   TTree*       mTreeKK;      //! X{g}
   TTree*       mTreeV0;      //! X{g}
   TTree*       mTreeGI;      //! X{g}

   TEveMCTrack        mK,  *mpK;    //!
   // MCTrackRef      mTR, *mpTR;   //!
   TEveHit            mH,  *mpH;    //!
   TEveCluster        mC,  *mpC;    //!
   TEveRecTrack       mR,  *mpR;    //!
   TEveRecKink        mKK, *mpKK;   //!
   TEveRecV0          mV0, *mpV0;   //!
   TEveMCRecCrossRef  mGI, *mpGI;   //!

public:
   TEveVSD(const Text_t* name="TEveVSD", const Text_t* title="");

   virtual void SetDirectory(TDirectory* dir);

   virtual void CreateTrees();
   virtual void DeleteTrees();

   virtual void CreateBranches();
   virtual void SetBranchAddresses();

   virtual void WriteTrees();
   virtual void LoadTrees();

   virtual void LoadVSD(const Text_t* vsd_file_name,
                        const Text_t* dir_name="Event0");

   static void DisableTObjectStreamersForVSDStruct();

   ClassDef(TEveVSD, 1); // Visualization Summary Data - a collection of trees holding standard event data in experiment independant format.
};

#endif
