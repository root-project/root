// @(#)root/net:$Id$
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
//    <grid>://<host>[:<port>], e.g. alien://alice.cern.ch              //
// Depending on the <grid> specified an appropriate plugin library      //
// will be loaded which will provide the real interface.                //
//                                                                      //
// Related classes are TGridResult.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

#include "TString.h"

#include "TGridJob.h"

class TGridResult;
class TGridJDL;
class TGridJob;
class TGridCollection;
class TGridJobStatusList;


class TGrid : public TObject {

protected:
   TString        fGridUrl; // the GRID url used to create the grid connection
   TString        fGrid;    // type of GRID (AliEn, ...)
   TString        fHost;    // GRID portal to which we are connected
   TString        fUser;    // user name
   TString        fPw;      // user passwd
   TString        fOptions; // options specified
   Int_t          fPort;    // port to which we are connected

public:
   TGrid() : fGridUrl(), fGrid(), fHost(), fUser(), fPw(), fOptions(), fPort(-1) { }
   virtual ~TGrid() { }

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
                                Bool_t /*interactive*/ = kFALSE,
                                UInt_t /*stream*/ = 2)
      { MayNotUse("Command"); return 0; }

   virtual TGridResult *Query(const char * /*path*/, const char * /*pattern*/,
                              const char * /*conditions*/ = "", const char * /*options*/ = "")
      { MayNotUse("Query"); return 0; }

   virtual TGridResult *LocateSites() { MayNotUse("LocalSites"); return 0; }

   //--- Catalogue Interface
   virtual TGridResult *Ls(const char* /*ldn*/ ="", Option_t* /*options*/ ="", Bool_t /*verbose*/ =kFALSE)
      { MayNotUse("Ls"); return 0; }
   virtual const char  *Pwd(Bool_t /*verbose*/ =kFALSE)
      { MayNotUse("Pwd"); return 0; }
   virtual const char  *GetHomeDirectory()
      { MayNotUse("GetHomeDirectory"); return 0; }
   virtual Bool_t Cd(const char* /*ldn*/ ="",Bool_t /*verbose*/ =kFALSE)
      { MayNotUse("Cd"); return kFALSE; }
   virtual Int_t  Mkdir(const char* /*ldn*/ ="", Option_t* /*options*/ ="", Bool_t /*verbose*/ =kFALSE)
      { MayNotUse("Mkdir"); return kFALSE; }
   virtual Bool_t Rmdir(const char* /*ldn*/ ="", Option_t* /*options*/ ="", Bool_t /*verbose*/ =kFALSE)
      { MayNotUse("Mkdir"); return kFALSE; }
   virtual Bool_t Register(const char* /*lfn*/ , const char* /*turl*/ , Long_t /*size*/ =-1, const char* /*se*/ =0, const char* /*guid*/ =0, Bool_t /*verbose*/ =kFALSE)
      { MayNotUse("Mkdir"); return kFALSE; }
   virtual Bool_t Rm(const char* /*lfn*/ , Option_t* /*option*/ ="", Bool_t /*verbose*/ =kFALSE)
      { MayNotUse("Mkdir"); return kFALSE; }

   //--- Job Submission Interface
   virtual TGridJob *Submit(const char * /*jdl*/)
      { MayNotUse("Submit"); return 0; }
   virtual TGridJDL *GetJDLGenerator()
      { MayNotUse("GetJDLGenerator"); return 0; }
   virtual TGridCollection *OpenCollection(const char *, UInt_t /*maxentries*/ = 1000000)
      { MayNotUse("OpenCollection"); return 0; }
   virtual TGridCollection *OpenCollectionQuery(TGridResult * /*queryresult*/,Bool_t /*nogrouping*/ = kFALSE)
      { MayNotUse("OpenCollection"); return 0; }
   virtual TGridJobStatusList* Ps(const char* /*options*/, Bool_t /*verbose*/ = kTRUE)
      { MayNotUse("Ps"); return 0; }
   virtual Bool_t KillById(TString /*jobid*/)
      { MayNotUse("KillById"); return kFALSE; }
   virtual Bool_t ResubmitById(TString /*jobid*/)
      { MayNotUse("ResubmitById"); return 0; }
   virtual Bool_t Kill(TGridJob *gridjob)
      { return ((gridjob)?KillById(gridjob->GetJobID()):kFALSE); }
   virtual Bool_t Resubmit(TGridJob* gridjob)
      { return ((gridjob)?ResubmitById(gridjob->GetJobID()):kFALSE); }

   //--- Load desired plugin and setup conection to GRID
   static TGrid *Connect(const char *grid, const char *uid = 0,
                         const char *pw = 0, const char *options = 0);

   ClassDef(TGrid,0)  // ABC defining interface to GRID services
};

R__EXTERN TGrid *gGrid;

#endif
