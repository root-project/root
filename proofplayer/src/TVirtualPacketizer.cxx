// @(#)root/proof:$Name:  $:$Id: TVirtualPacketizer.cxx,v 1.1 2002/07/17 12:29:37 rdm Exp $
// Author: Maarten Ballintijn    9/7/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualPacketizer                                                   //
//                                                                      //
// XXX update Comment XXX                                               //
// This class generates packets to be processed on PROOF slave servers. //
// A packet is an event range (begin entry and number of entries) or    //
// object range (first object and number of objects) in a TTree         //
// (entries) or a directory (objects) in a file.                        //
// Packets are generated taking into account the performance of the     //
// remote machine, the time it took to process a previous packet on     //
// the remote machine, the locality of the database files, etc.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TVirtualPacketizer.h"
#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include "TDSet.h"


ClassImp(TVirtualPacketizer)

//______________________________________________________________________________
TVirtualPacketizer::TVirtualPacketizer()
{
   fValid = kTRUE;
}

//______________________________________________________________________________
TVirtualPacketizer::~TVirtualPacketizer()
{
}

//______________________________________________________________________________
Long64_t TVirtualPacketizer::GetEntries(Bool_t tree, TDSetElement *e)
{
   Long64_t entries;
   TFile *file = TFile::Open(e->GetFileName());

   if ( file->IsZombie() ) {
      Error("GetEntries","Cannot open file: %s (%s)",
            e->GetFileName(), strerror(file->GetErrno()) );
      return -1;
   }

   TDirectory *dirsave = gDirectory;
   if ( ! file->cd(e->GetDirectory()) ) {
      Error("GetEntries","Cannot cd to: %s", e->GetDirectory() );
      delete file;
      return -1;
   }
   TDirectory *dir = gDirectory;
   dirsave->cd();

   if ( tree ) {
      TKey *key = dir->GetKey(e->GetObjName());
      if ( key == 0 ) {
         Error("GetEntries","Cannot find tree \"%s\" in %s",
               e->GetObjName(), e->GetFileName() );
         delete file;
         return -1;
      }
      TTree *tree = (TTree *) key->ReadObj();
      if ( tree == 0 ) {
         // Error always reported?
         delete file;
         return -1;
      }
      entries = (Long64_t) tree->GetEntries();
      delete tree;

   } else {
      TList *keys = dir->GetListOfKeys();
      entries = keys->GetSize();
   }

   delete file;

   return entries;
}


//______________________________________________________________________________
Long64_t TVirtualPacketizer::GetEntriesProcessed() const
{
   AbstractMethod("GetEntriesProcessed");

   return 0;
}


//______________________________________________________________________________
Long64_t TVirtualPacketizer::GetEntriesProcessed(TSlave *sl) const
{
   AbstractMethod("GetEntriesProcessed(TSlave *sl)");

   return 0;
}


//______________________________________________________________________________
TDSetElement *TVirtualPacketizer::GetNextPacket(TSlave *sl)
{
   AbstractMethod("GetNextPacket");

   return 0;
}
