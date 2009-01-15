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
#include "TFile.h"

//==============================================================================
// TEveVSD
//==============================================================================

//______________________________________________________________________________
//
// Visualization Summary Data - a collection of trees holding standard
// event data in experiment independant format.

ClassImp(TEveVSD);

//______________________________________________________________________________
TEveVSD::TEveVSD(const char* , const char* ) :
   TObject(),

   fBuffSize  (128*1024),

   fFile      (0),
   fDirectory (0),

   fTreeK  (0),
   fTreeH  (0),
   fTreeC  (0),
   fTreeR  (0),
   fTreeKK (0),
   fTreeV0 (0),
   fTreeCC (0),
   fTreeGI (0),

   fK(),  fpK (&fK),
   fH(),  fpH (&fH),
   fC(),  fpC (&fC),
   fR(),  fpR (&fR),
   fKK(), fpKK(&fKK),
   fV0(), fpV0(&fV0),
   fCC(), fpCC(&fCC),
   fGI(), fpGI(&fGI)
{
   // Constructor.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveVSD::SetDirectory(TDirectory* dir)
{
   // Set directory in which the trees are (or will be) created.
   fDirectory = dir;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveVSD::CreateTrees()
{
   // Create internal trees.

   fDirectory->cd();
   fTreeK  = new TTree("Kinematics", "Simulated tracks.");
   fTreeH  = new TTree("Hits",       "Combined detector hits.");
   fTreeC  = new TTree("Clusters",   "Reconstructed clusters.");
   fTreeR  = new TTree("RecTracks",  "Reconstructed tracks.");
   fTreeKK = new TTree("RecKinks",   "Reconstructed kinks.");
   fTreeV0 = new TTree("RecV0s",     "Reconstructed V0s.");
   fTreeCC = new TTree("RecCascades","Reconstructed cascades.");
   fTreeGI = new TTree("TEveMCRecCrossRef",    "Objects prepared for cross query.");
}

//______________________________________________________________________________
void TEveVSD::DeleteTrees()
{
   // Delete interal trees.

   delete fTreeK;      fTreeK      = 0;
   delete fTreeH;      fTreeH      = 0;
   delete fTreeC;      fTreeC      = 0;
   delete fTreeR;      fTreeR      = 0;
   delete fTreeV0;     fTreeV0     = 0;
   delete fTreeKK;     fTreeKK     = 0;
   delete fTreeGI;     fTreeGI     = 0;
}

//______________________________________________________________________________
void TEveVSD::CreateBranches()
{
   // Create internal VSD branches.

   if(fTreeK)
      fTreeK ->Branch("K",  "TEveMCTrack",  &fpK,  fBuffSize);
   if(fTreeH)
      fTreeH ->Branch("H",  "TEveHit",      &fpH,  fBuffSize);
   if(fTreeC)
      fTreeC ->Branch("C",  "TEveCluster",  &fpC,  fBuffSize);
   if(fTreeR)
      fTreeR ->Branch("R",  "TEveRecTrack", &fpR,  fBuffSize);
   if(fTreeKK)
      fTreeKK->Branch("KK", "TEveRecKink",  &fpKK, fBuffSize);
   if(fTreeV0)
      fTreeV0->Branch("V0", "TEveRecV0",    &fpV0, fBuffSize);

   if(fTreeGI) {
      fTreeGI->Branch("GI", "TEveMCRecCrossRef",  &fpGI, fBuffSize);
      fTreeGI->Branch("K.", "TEveMCTrack",  &fpK);
      fTreeGI->Branch("R.", "TEveRecTrack", &fpR);
   }
}

//______________________________________________________________________________
void TEveVSD::SetBranchAddresses()
{
   // Set branche addresses of internal trees.

   if(fTreeK)
      fTreeK ->SetBranchAddress("K",  &fpK);
   if(fTreeH)
      fTreeH ->SetBranchAddress("H",  &fpH);
   if(fTreeC)
      fTreeC ->SetBranchAddress("C",  &fpC);
   if(fTreeR)
      fTreeR ->SetBranchAddress("R",  &fpR);
   if(fTreeKK)
      fTreeKK->SetBranchAddress("KK", &fpKK);
   if(fTreeV0)
      fTreeV0->SetBranchAddress("V0", &fpV0);

   if(fTreeGI) {
      fTreeGI->SetBranchAddress("GI", &fpGI);
      fTreeGI->SetBranchAddress("K.", &fpK);
      fTreeGI->SetBranchAddress("R.", &fpR);
   }
}

//______________________________________________________________________________
void TEveVSD::WriteTrees()
{
   // Does nothing here ... reimplemented in sub-classes.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveVSD::LoadTrees()
{
   // Load internal trees from directory.

   static const TEveException eH("TEveVSD::LoadTrees ");

   if(fDirectory == 0)
      throw(eH + "directory not set.");

   printf("Reading kinematics.\n");
   fTreeK = (TTree*) fDirectory->Get("Kinematics");
   if(fTreeK == 0) {
      printf("%s Kinematics not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   printf("Reading hits.\n");
   fTreeH = (TTree*) fDirectory->Get("Hits");
   if(fTreeH == 0) {
      printf("%s Hits not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   printf("Reading clusters.\n");
   fTreeC = (TTree*) fDirectory->Get("Clusters");
   if(fTreeC == 0) {
      printf("%s Clusters not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   printf("Reading reconstructed tracks.\n");
   fTreeR = (TTree*) fDirectory->Get("RecTracks");
   if(fTreeR == 0) {
      printf("%s RecTracks not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   printf("Reading reconstructed kinks. \n");
   fTreeKK =  (TTree*) fDirectory->Get("RecKinks");
   if(fTreeKK == 0) {
      printf("%s Kinks not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   printf("Reading Reconstructed V0s.\n");
   fTreeV0 =  (TTree*) fDirectory->Get("RecV0s");
   if(fTreeV0 == 0) {
      printf("%s V0 not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

   printf("Reading TEveMCRecCrossRef.\n");
   fTreeGI = (TTree*)fDirectory->Get("TEveMCRecCrossRef");
   if(fTreeGI == 0) {
      printf("%s TEveMCRecCrossRef not available in fDirectory %s.\n",
             eH.Data(), fDirectory->GetName());
   }

}

//______________________________________________________________________________
void TEveVSD::LoadVSD(const char* vsd_file_name, const char* dir_name)
{
   // Load VSD data from given file and directory.

   static const TEveException eH("TEveVSD::LoadVSD ");

   fFile = TFile::Open(vsd_file_name);
   if( fFile == 0)
      throw(eH + "can not open TEveVSD file '" + vsd_file_name + "'.");

   fDirectory = (TDirectory*) fFile->Get(dir_name);
   if (fDirectory == 0)
      throw(eH + "directory '" + dir_name + "' not found in TEveVSD file '" + vsd_file_name + "'.");
   printf("%p\n", (void*)fDirectory);
   LoadTrees();
   SetBranchAddresses();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveVSD::DisableTObjectStreamersForVSDStruct()
{
   // Disble TObject streamers for those VSD structs that inherit from
   // TObject directly.

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
