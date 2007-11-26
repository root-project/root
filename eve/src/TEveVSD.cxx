// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveVSD.h>
#include <TFile.h>

//______________________________________________________________________________
// TEveVSD
//
// Visualization Summary Data - a collection of trees holding standard event data in experiment independant format.

ClassImp(TEveVSD)

//______________________________________________________________________________
TEveVSD::TEveVSD(const Text_t* , const Text_t* ) :
   TObject(),

   fBuffSize  (128*1024),

   mFile      (0),
   mDirectory (0),

   mTreeK  (0),
   //mTreeTR (0),
   mTreeH  (0),
   mTreeC  (0),
   mTreeR  (0),
   mTreeKK (0),
   mTreeV0 (0),
   mTreeGI (0),

   mK(),  mpK (&mK),
   mH(),  mpH (&mH),
   mC(),  mpC (&mC),
   mR(),  mpR (&mR),
   mKK(), mpKK(&mKK),
   mV0(), mpV0(&mV0),
   mGI(), mpGI(&mGI)
{}

/******************************************************************************/

//______________________________________________________________________________
void TEveVSD::SetDirectory(TDirectory* dir)
{
   mDirectory = dir;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveVSD::CreateTrees()
{
   mDirectory->cd();
   // TR missing ...
   mTreeK  = new TTree("Kinematics", "Simulated tracks.");
   mTreeH  = new TTree("Hits",       "Combined detector hits.");
   mTreeC  = new TTree("Clusters",   "Reconstructed clusters.");
   mTreeR  = new TTree("RecTracks",  "Reconstructed tracks.");
   mTreeKK = new TTree("RecKinks",   "Reconstructed kinks.");
   mTreeV0 = new TTree("RecV0s",     "Reconstructed V0s.");
   mTreeGI = new TTree("TEveMCRecCrossRef",    "Objects prepared for cross query.");
}

//______________________________________________________________________________
void TEveVSD::DeleteTrees()
{
   delete mTreeK;      mTreeK      = 0;
   // delete mTreeTR;     mTreeTR     = 0;
   delete mTreeH;      mTreeH      = 0;
   delete mTreeC;      mTreeC      = 0;
   delete mTreeR;      mTreeR      = 0;
   delete mTreeV0;     mTreeV0     = 0;
   delete mTreeKK;     mTreeKK     = 0;
   delete mTreeGI;     mTreeGI     = 0;
}

//______________________________________________________________________________
void TEveVSD::CreateBranches()
{
   // TR missing ...
   if(mTreeK)
      mTreeK ->Branch("K",  "TEveMCTrack",  &mpK,  fBuffSize);
   if(mTreeH)
      mTreeH ->Branch("H",  "TEveHit",      &mpH,  fBuffSize);
   if(mTreeC)
      mTreeC ->Branch("C",  "TEveCluster",  &mpC,  fBuffSize);
   if(mTreeR)
      mTreeR ->Branch("R",  "TEveRecTrack", &mpR,  fBuffSize);
   if(mTreeKK)
      mTreeKK->Branch("KK", "TEveRecKink",  &mpKK, fBuffSize);
   if(mTreeV0)
      mTreeV0->Branch("V0", "TEveRecV0",    &mpV0, fBuffSize);

   if(mTreeGI) {
      mTreeGI->Branch("GI", "TEveMCRecCrossRef",  &mpGI, fBuffSize);
      mTreeGI->Branch("K.", "TEveMCTrack",  &mpK);
      mTreeGI->Branch("R.", "TEveRecTrack", &mpR);
   }
}

//______________________________________________________________________________
void TEveVSD::SetBranchAddresses()
{
   // TR missing ...
   if(mTreeK)
      mTreeK ->SetBranchAddress("K",  &mpK);
   if(mTreeH)
      mTreeH ->SetBranchAddress("H",  &mpH);
   if(mTreeC)
      mTreeC ->SetBranchAddress("C",  &mpC);
   if(mTreeR)
      mTreeR ->SetBranchAddress("R",  &mpR);
   if(mTreeKK)
      mTreeKK->SetBranchAddress("KK", &mpKK);
   if(mTreeV0)
      mTreeV0->SetBranchAddress("V0", &mpV0);

   if(mTreeGI) {
      mTreeGI->SetBranchAddress("GI", &mpGI);
      mTreeGI->SetBranchAddress("K.", &mpK);
      mTreeGI->SetBranchAddress("R.", &mpR);
   }
}

//______________________________________________________________________________
void TEveVSD::WriteTrees()
{
   // Does nothing here ...
}

/******************************************************************************/

//______________________________________________________________________________
void TEveVSD::LoadTrees()
{
   static const TEveException eH("TEveVSD::LoadTrees ");

   if(mDirectory == 0)
      throw(eH + "directory not set.");

   printf("Reading kinematics.\n");
   mTreeK = (TTree*) mDirectory->Get("Kinematics");
   if(mTreeK == 0) {
      printf("%s Kinematics not available in mDirectory %s.\n",
             eH.Data(), mDirectory->GetName());
   }

   printf("Reading hits.\n");
   mTreeH = (TTree*) mDirectory->Get("Hits");
   if(mTreeH == 0) {
      printf("%s Hits not available in mDirectory %s.\n",
             eH.Data(), mDirectory->GetName());
   }

   printf("Reading clusters.\n");
   mTreeC = (TTree*) mDirectory->Get("Clusters");
   if(mTreeC == 0) {
      printf("%s Clusters not available in mDirectory %s.\n",
             eH.Data(), mDirectory->GetName());
   }

   printf("Reading reconstructed tracks.\n");
   mTreeR = (TTree*) mDirectory->Get("RecTracks");
   if(mTreeR == 0) {
      printf("%s RecTracks not available in mDirectory %s.\n",
             eH.Data(), mDirectory->GetName());
   }

   printf("Reading reconstructed kinks. \n");
   mTreeKK =  (TTree*) mDirectory->Get("RecKinks");
   if(mTreeKK == 0) {
      printf("%s Kinks not available in mDirectory %s.\n",
             eH.Data(), mDirectory->GetName());
   }

   printf("Reading Reconstructed V0s.\n");
   mTreeV0 =  (TTree*) mDirectory->Get("RecV0s");
   if(mTreeV0 == 0) {
      printf("%s V0 not available in mDirectory %s.\n",
             eH.Data(), mDirectory->GetName());
   }

   printf("Reading TEveMCRecCrossRef.\n");
   mTreeGI = (TTree*)mDirectory->Get("TEveMCRecCrossRef");
   if(mTreeGI == 0) {
      printf("%s TEveMCRecCrossRef not available in mDirectory %s.\n",
             eH.Data(), mDirectory->GetName());
   }

}

//______________________________________________________________________________
void TEveVSD::LoadVSD(const Text_t* vsd_file_name, const Text_t* dir_name)
{
   static const TEveException eH("TEveVSD::LoadVSD ");

   mFile = TFile::Open(vsd_file_name);
   if(mFile == 0)
      throw(eH + "can not open TEveVSD file '" + vsd_file_name + "'.");

   mDirectory = (TDirectory*) mFile->Get(dir_name);
   if(mDirectory == 0)
      throw(eH + "directory '" + dir_name + "' not found in TEveVSD file '" + vsd_file_name + "'.");
   printf("%p\n", (void*)mDirectory);
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
