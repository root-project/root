// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveVSD.h"

/** \class TEveVSD
\ingroup TEve
Visualization Summary Data - a collection of trees holding standard
event data in experiment independent format.
*/


////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveVSD::TEveVSD(const char* , const char*) :
   TObject(),

   fFile      (nullptr),
   fDirectory (nullptr),
   fBuffSize  (32000),
   fVerbose   (0),

   fTreeK  (nullptr),
   fTreeH  (nullptr),
   fTreeC  (nullptr),
   fTreeR  (nullptr),
   fTreeKK (nullptr),
   fTreeV0 (nullptr),
   fTreeCC (nullptr),
   fTreeGI (nullptr),

   fK(),  fpK (&fK),
   fH(),  fpH (&fH),
   fC(),  fpC (&fC),
   fR(),  fpR (&fR),
   fKK(), fpKK(&fKK),
   fV0(), fpV0(&fV0),
   fCC(), fpCC(&fCC),
   fGI(), fpGI(&fGI)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveVSD::~TEveVSD()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set directory in which the trees are (or will be) created.

void TEveVSD::SetDirectory(TDirectory* dir)
{
   fDirectory = dir;
}

////////////////////////////////////////////////////////////////////////////////
/// Create internal trees.

void TEveVSD::CreateTrees()
{
   fDirectory->cd();
   fTreeK  = new TTree("Kinematics", "Simulated tracks.");
   fTreeH  = new TTree("Hits",       "Combined detector hits.");
   fTreeC  = new TTree("Clusters",   "Reconstructed clusters.");
   fTreeR  = new TTree("RecTracks",  "Reconstructed tracks.");
   fTreeKK = new TTree("RecKinks",   "Reconstructed kinks.");
   fTreeV0 = new TTree("RecV0s",     "Reconstructed V0s.");
   fTreeCC = new TTree("RecCascades","Reconstructed cascades.");
   fTreeGI = new TTree("TEveMCRecCrossRef", "Objects prepared for cross query.");
}

////////////////////////////////////////////////////////////////////////////////
/// Delete internal trees.

void TEveVSD::DeleteTrees()
{
   delete fTreeK;  fTreeK  = nullptr;
   delete fTreeH;  fTreeH  = nullptr;
   delete fTreeC;  fTreeC  = nullptr;
   delete fTreeR;  fTreeR  = nullptr;
   delete fTreeV0; fTreeV0 = nullptr;
   delete fTreeKK; fTreeKK = nullptr;
   delete fTreeGI; fTreeGI = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create internal VSD branches.

void TEveVSD::CreateBranches()
{
   if (fTreeK)  fTreeK ->Branch("K",  "TEveMCTrack",  &fpK);
   if (fTreeH)  fTreeH ->Branch("H",  "TEveHit",      &fpH);
   if (fTreeC)  fTreeC ->Branch("C",  "TEveCluster",  &fpC);
   if (fTreeR)  fTreeR ->Branch("R",  "TEveRecTrack", &fpR);
   if (fTreeKK) fTreeKK->Branch("KK", "TEveRecKink",  &fpKK);
   if (fTreeV0) fTreeV0->Branch("V0", "TEveRecV0",    &fpV0);

   if (fTreeGI)
   {
      fTreeGI->Branch("GI", "TEveMCRecCrossRef",  &fpGI);
      fTreeGI->Branch("K.", "TEveMCTrack",  &fpK);
      fTreeGI->Branch("R.", "TEveRecTrack", &fpR);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set branche addresses of internal trees.

void TEveVSD::SetBranchAddresses()
{
   if (fTreeK)  fTreeK ->SetBranchAddress("K",  &fpK);
   if (fTreeH)  fTreeH ->SetBranchAddress("H",  &fpH);
   if (fTreeC)  fTreeC ->SetBranchAddress("C",  &fpC);
   if (fTreeR)  fTreeR ->SetBranchAddress("R",  &fpR);
   if (fTreeKK) fTreeKK->SetBranchAddress("KK", &fpKK);
   if (fTreeV0) fTreeV0->SetBranchAddress("V0", &fpV0);

   if (fTreeGI)
   {
      fTreeGI->SetBranchAddress("GI", &fpGI);
      fTreeGI->SetBranchAddress("K.", &fpK);
      fTreeGI->SetBranchAddress("R.", &fpR);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Does nothing here ... reimplemented in sub-classes.

void TEveVSD::WriteTrees()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Load internal trees from directory.

void TEveVSD::LoadTrees()
{
   static const TEveException eH("TEveVSD::LoadTrees ");

   if (fDirectory == nullptr)
      throw eH + "directory not set.";

   fTreeK = (TTree*) fDirectory->Get("Kinematics");
   if (fTreeK == nullptr && fVerbose) {
      printf("%s Kinematics not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   fTreeH = (TTree*) fDirectory->Get("Hits");
   if (fTreeH == nullptr && fVerbose) {
      printf("%s Hits not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   fTreeC = (TTree*) fDirectory->Get("Clusters");
   if (fTreeC == nullptr && fVerbose) {
      printf("%s Clusters not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   fTreeR = (TTree*) fDirectory->Get("RecTracks");
   if (fTreeR == nullptr && fVerbose) {
      printf("%s RecTracks not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   fTreeKK =  (TTree*) fDirectory->Get("RecKinks");
   if (fTreeKK == nullptr && fVerbose) {
      printf("%s Kinks not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   fTreeV0 =  (TTree*) fDirectory->Get("RecV0s");
   if (fTreeV0 == nullptr && fVerbose) {
      printf("%s V0 not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   fTreeGI = (TTree*)fDirectory->Get("TEveMCRecCrossRef");
   if(fTreeGI == nullptr && fVerbose) {
      printf("%s TEveMCRecCrossRef not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Disable TObject streamers for those VSD structs that inherit from
/// TObject directly.

void TEveVSD::DisableTObjectStreamersForVSDStruct()
{
   // TEveVector is not TObject

   // TEveMCTrack derives from TParticle
   TParticle::Class()->IgnoreTObjectStreamer(true);

   TEveHit::Class()->IgnoreTObjectStreamer(true);
   TEveCluster::Class()->IgnoreTObjectStreamer(true);

   TEveRecTrack::Class()->IgnoreTObjectStreamer(true);
   // TEveRecKink derives from TEveRecTrack

   TEveRecV0::Class()->IgnoreTObjectStreamer(true);

   TEveMCRecCrossRef::Class()->IgnoreTObjectStreamer(true);
}
