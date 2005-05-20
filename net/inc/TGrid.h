// @(#)root/net:$Name:  $:$Id: TGrid.h,v 1.11 2005/05/13 08:49:54 rdm Exp $
// Author: Fons Rademakers   3/1/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGrid
#define ROOT_TGrid

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGrid                                                                //
//                                                                      //
// Abstract base class defining interface to common GRID services.      //
//                                                                      //
// To open a connection to a GRID use the static method Connect().      //
// The argument of Connect() is of the form:                            //
//    <grid>://<host>[:<port>], e.g.                                    //
// alien://alice.cern.ch, globus://glsvr1.cern.ch, ...                  //
// Depending on the <grid> specified an appropriate plugin library      //
// will be loaded which will provide the real interface.                //
//                                                                      //
// Related classes are TGridResult.                                     //
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

class TGridResult;
class TGridJDL;
class TGridJob;
class TDSet;
class TFile;
class TDirectory;
class TList;


class TGrid : public TObject {

protected:
   TString        fGridUrl; // the GRID url used to create the grid connection
   TString        fGrid;    // type of GRID (AliEn, Globus, ...)
   TString        fHost;    // GRID portal to which we are connected
   TString        fUser;    // user name
   TString        fPw;      // user passwd
   TString        fOptions; // options specified
   Int_t          fPort;    // port to which we are connected
   TStopwatch     fWatch;   // stop watch to measure file copy speed
   TList         *fMergerFileList;        // a list of files, which shall be merged
   TFile         *fMergerOutputFile;      // the outputfile for merging
   TString        fMergerOutputFilename;  // the name of the outputfile for merging
   TString        fMergerOutputFilename1; // the name of the temporary outputfile for merging

   void           PrintProgress(Long64_t bytesread, Long64_t size);

public:
   TGrid();
   virtual ~TGrid();

   const char    *GridUrl() const { return fGridUrl; }
   const char    *GetGrid() const { return fGrid; }
   const char    *GetHost() const { return fHost; }
   const char    *GetUser() const { return fUser; }
   const char    *GetPw() const { return fPw; }
   const char    *GetOptions() const { return fOptions; }
   Int_t          GetPort() const { return fPort; }
   virtual Bool_t IsConnected() const { return fPort == -1 ? kFALSE : kTRUE; }

   virtual void Shell() { MayNotUse("Shell"); }
   virtual void Stdout() { MayNotUse("Stdout"); }
   virtual void Stderr() { MayNotUse("Stderr"); }

   virtual TGridResult *Command(const char * /*command*/,
                                Bool_t /*interactive*/ = kFALSE)
      { MayNotUse("Command"); return 0; }

   virtual TGridResult *Query(const char * /*path*/, const char * /*pattern*/,
                              const char * /*conditions*/, const char * /*options*/)
      { MayNotUse("Query"); return 0; }

   virtual TGridResult *LocateSites() { MayNotUse("LocalSites"); return 0; }

   virtual Bool_t Query2Dataset(TDSet * /*dset*/, const char * /*path*/,
                                const char * /*pattern*/,
                                const char * /*conditions*/,
                                const char * /*options*/)
      { MayNotUse("Query2Dataset"); return kFALSE; }

   //--- file management interface
   virtual Bool_t Cp(const char *src, const char *dst, Bool_t progressbar = kTRUE,
                     UInt_t buffersize = 1000000);
   virtual Bool_t SetCWD(const char * /*path*/) { MayNotUse("SetCWD"); return kFALSE; }
   virtual const char *GetCWD() { MayNotUse("GetCWD"); return 0; }

   //--- file merging interface
   virtual void   MergerReset();
   virtual Bool_t MergerAddFile(const char *url);
   virtual Bool_t MergerOutputFile(const char *url);
   virtual void   MergerPrintFiles(Option_t *options);
   virtual Bool_t MergerMerge();
   virtual Bool_t MergerMergeRecursive(TDirectory *target, TList *sourcelist);

   //--- job Submission Interface
   virtual TGridJob *Submit(const char * /*jdl*/) { MayNotUse("Submit"); return 0; }
   virtual TGridJDL *GetJDLGenerator() { MayNotUse("GetJDLGenerator"); return 0; }

   //--- load desired plugin and setup conection to GRID
   static TGrid *Connect(const char *grid, const char *uid = 0,
                         const char *pw = 0, const char *options = 0);

   ClassDef(TGrid,0)  // ABC defining interface to GRID services
};

R__EXTERN TGrid *gGrid;

#endif
