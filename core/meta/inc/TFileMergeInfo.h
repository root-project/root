// @(#)root/proofplayer:$Id$
// Author: Philippe Canal May, 2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileMergeInfo
#define ROOT_TFileMergeInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileMergeInfo                                                       //
//                                                                      //
// This class helps passing information from the TFileMerger to         //
// the objects being merged.                                            //
//                                                                      //
// It provides access to the output directory pointer (fOutputDirectory)//
// to whether or not this is the first time Merge is being called in the//
// serie (for example for TTree, the first time we also need to Clone   //
// the object on which Merge is called), and provides for a User Data   //
// object to be passed along to each of the calls to Merge.             //
// The fUserData object is owned by the TFileMergeInfo and will be      //
// deleted when the TFileMerger moves on to the next set of objects.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

class TDirectory;

class TFileMergeInfo {
public:
   TDirectory  *fOutputDirectory;  // Target directory where the merged object will be written.
   Bool_t       fIsFirst;          // True if this is the first call to Merge for this series of object.
   TString      fOptions;          // Additional text based option being passed down to customize the merge.
   TObject     *fUserData;         // Place holder to pass extra information.  This object will be deleted at the end of each series of objects.
 
   TFileMergeInfo(TDirectory *outputfile) : fOutputDirectory(outputfile), fIsFirst(kTRUE), fUserData(0) {}
   virtual ~TFileMergeInfo() { delete fUserData; } ;
   
   void Reset() { fIsFirst = kTRUE; delete fUserData; fUserData = 0; }
   
   ClassDef(TFileMergeInfo, 0);
};

#endif
