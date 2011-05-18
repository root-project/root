// @(#)root/alien:$Id$
// Author: Andreas Peters   5/5/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlien
#define ROOT_TAlien


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlien                                                               //
//                                                                      //
// Class defining interface to TAlien GRID services.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGrid
#include "TGrid.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

class GapiUI;
class TGridJob;
class TGridJobStatusList;
class TMap;


class TAlien : public TGrid {

public:
   enum { kSTDOUT = 0, kSTDERR = 1 , kOUTPUT = 2, kENVIR = 3 };
   enum CatalogType { kFailed = -1, kFile = 0, kDirectory, kCollection };

private:
   // Stream content types.
   // The streams are originally defined in the CODEC.h of
   // the external gapiUI library.

   GapiUI    *fGc;    // the GapiUI object implementing the communication layer
   TString    fPwd;   // working directory
   TString    fHome;  // home directory with alien:// prefix

   TGridResult         *Command(const char *command, bool interactive = kFALSE,
                                UInt_t stream = kOUTPUT);
   virtual TGridResult *Query(const char *path, const char *pattern,
                              const char *conditions = "", const char *options = "");

   virtual TGridResult *LocateSites();
   virtual TGridResult *OpenDataset(const char *lfn, const char *options = "");

public:
   TAlien(const char *gridurl, const char *uid=0, const char *passwd=0,
          const char *options=0);
   virtual ~TAlien();

   void Shell();           // start an interactive ALIEN shell

   void Stdout();          // print the stdout of the last executed command
   void Stderr();          // print the stderr of the last executed command

   TMap  *GetColumn(UInt_t stream=0, UInt_t column=0);
   UInt_t GetNColumns(UInt_t stream);

   const char *GetStreamFieldValue(UInt_t stream, UInt_t column, UInt_t row);
   const char *GetStreamFieldKey(UInt_t stream, UInt_t column, UInt_t row);

   TString Escape(const char *input);
   virtual TGridJob *Submit(const char *jdl); // submit a grid job
   virtual TGridJDL *GetJDLGenerator();       // get a AliEn grid JDL object
   virtual TGridCollection* OpenCollection(const char* collectionfile, UInt_t maxentries = kTRUE);
   virtual TGridCollection* OpenCollectionQuery(TGridResult * queryresult, Bool_t nogrouping = kFALSE);
   virtual TGridJobStatusList* Ps(const char* options, Bool_t verbose = kTRUE);
   virtual Bool_t KillById(TString jobid);
   virtual Bool_t ResubmitById(TString jobid);

   //--- Catalogue Interface
   virtual TGridResult *Ls(const char *ldn="", Option_t *options="", Bool_t verbose=kFALSE);
   virtual const char  *Pwd(Bool_t verbose=kFALSE);
   virtual const char  *GetHomeDirectory() { return fHome.Data(); }
   virtual Bool_t Cd(const char *ldn="", Bool_t verbose=kFALSE);
   virtual Int_t Mkdir(const char *ldn="", Option_t *options="", Bool_t verbose=kFALSE);
   virtual Bool_t Rmdir(const char *ldn="", Option_t *options="", Bool_t verbose=kFALSE);
   virtual Bool_t Register(const char *lfn, const char *turl, Long_t size=-1,
                           const char *se=0, const char *guid=0, Bool_t verbose=kFALSE);
   virtual Bool_t Rm(const char *lfn, Option_t *option="", Bool_t verbose=kFALSE);
   virtual CatalogType Type(const char* lfn, Option_t* option = "", Bool_t verbose=kFALSE);
   virtual TGridResult* GetCollection(const char* lfn, Option_t* option = "", Bool_t verbose=kFALSE);

   //--- Software Packages
   virtual TGridResult* ListPackages(const char* alienpackagedir="/alice/packages");
   ClassDef(TAlien,0)  // Interface to Alien GRID services
};

#endif
