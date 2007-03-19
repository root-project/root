// @(#)root/proofplayer:$Name:  $:$Id: TFileMerger.h,v 1.3 2007/02/09 11:51:09 rdm Exp $
// Author: Andreas Peters + Fons Rademakers   26/5/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileMerger
#define ROOT_TFileMerger

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileMerger                                                          //
//                                                                      //
// This class provides file copy and merging services.                  //
//                                                                      //
// It can be used to copy files (not only ROOT files), using TFile or   //
// any of its remote file access plugins. It is therefore usefull in    //
// a Grid environment where the files might be accessable via Castor,   //
// rfio, dcap, etc.                                                     //
// The merging interface allows files containing histograms and trees   //
// to be merged, like the standalone hadd program.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TStopwatch
#include "TStopwatch.h"
#endif

class TList;
class TFile;
class TDirectory;


class TFileMerger : public TObject {
private:
   TFileMerger(const TFileMerger&); // Not implemented
   TFileMerger& operator=(const TFileMerger&); // Not implemented

protected:
   TStopwatch     fWatch;           // stop watch to measure file copy speed
   TList         *fFileList;        // a list of files, which shall be merged
   TFile         *fOutputFile;      // the outputfile for merging
   TString        fOutputFilename;  // the name of the outputfile for merging
   TString        fOutputFilename1; // the name of the temporary outputfile for merging

   void           PrintProgress(Long64_t bytesread, Long64_t size);

public:
   TFileMerger();
   virtual ~TFileMerger();

   const char *GetOutputFileName() const { return fOutputFilename; }

    //--- file management interface
   virtual Bool_t Cp(const char *src, const char *dst, Bool_t progressbar = kTRUE,
                     UInt_t buffersize = 1000000);
   virtual Bool_t SetCWD(const char * /*path*/) { MayNotUse("SetCWD"); return kFALSE; }
   virtual const char *GetCWD() { MayNotUse("GetCWD"); return 0; }

   //--- file merging interface
   virtual void   Reset();
   virtual Bool_t AddFile(const char *url);
   virtual Bool_t OutputFile(const char *url);
   virtual void   PrintFiles(Option_t *options);
   virtual Bool_t Merge();
   virtual Bool_t MergeRecursive(TDirectory *target, TList *sourcelist);

   ClassDef(TFileMerger,1)  // File copying and merging services
};

#endif
