// @(#)root/net:$Name:  $:$Id: TGrid.h,v 1.7 2003/11/13 17:01:15 rdm Exp $
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

#if !defined(__CINT__)
#include <sys/stat.h>
#endif

#include <string>
#ifdef R__GLOBALSTL
namespace std { using ::string; }
#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TGridResult;
class TGridProof;

typedef Int_t   Grid_JobId_t;
typedef ULong_t Grid_ResultHandle_t;
typedef Long_t  Grid_ProofSessionStatus_t;
typedef ULong_t Grid_FileHandle_t;
typedef ULong_t Grid_ResultHandle_t;
typedef ULong_t Grid_FileHandle_t;

struct Grid_Result_t {
   std::string         name;
   std::string         name2;
   struct stat         info;
   Grid_ResultHandle_t data;

   Grid_Result_t() {
      memset(&info, 0, sizeof(info));
      data = 0;
   }
};

typedef void          Grid_AccessPath_t;
typedef void          Grid_JobStatus_t;
typedef Grid_Result_t Grid_FileEntry_t;
typedef void          Grid_ProofSession_t;


class TGrid : public TObject {

protected:
   TString        fGrid;   // type of GRID (AliEn, Globus, ...)
   TString        fHost;   // GRID portal to which we are connected
   Int_t          fPort;   // port to which we are connected
   TGridProof    *fProof;  // PROOF pointer

   TGrid() : fPort(-1), fProof(0) { }

public:
#ifndef __CINT__
   typedef struct stat gridstat_t;
#else
   struct gridstat_t;
#endif

   virtual ~TGrid() { }

   const char    *GetGrid() const { return fGrid; }
   const char    *GetHost() const { return fHost; }
   Int_t          GetPort() const { return fPort; }
   virtual Bool_t IsConnected() const { return fPort == -1 ? kFALSE : kTRUE; }

   virtual const char *GetUser() = 0;

   //--- file catalog browsing
   virtual Grid_ResultHandle_t OpenDir(const char *ldn) = 0;

   virtual Grid_Result_t *ReadResult(Grid_ResultHandle_t hResult) = 0;
   virtual void CloseResult(Grid_ResultHandle_t hResult) = 0;
   virtual void ResetResult(Grid_ResultHandle_t hResult) = 0;

   //--- file catalog management
   virtual Int_t Mkdir(const char *ldn, Bool_t recursive = kFALSE) = 0;
   virtual Int_t Rmdir(const char *dir, Bool_t recursive = kFALSE) = 0;
   virtual Int_t Rm(const char *lfn, Bool_t recursive = kFALSE) = 0;
   virtual Int_t Cp(const char *sourcelfn, const char *targetlfn) = 0;
   virtual Int_t Mv(const char *sourcelfn, const char *targetlfn) = 0;
   virtual Int_t Chmod(const char *lfn, UInt_t mode) = 0;
   virtual Int_t Chown(const char *lfn, const char *owner,
                       const char *group) = 0;

   virtual Int_t AddFile(const char *newlfn, const char *pfn, Int_t size = -1,
                         const char *msn = 0, char *guid = 0) = 0;
   virtual Int_t AddFileMirror(const char *lfn, const char *pfn,
                               const char *msn) = 0;
   virtual Int_t RegisterFile(const char *lfn, const char *pfn,
                              const char *msn = "") = 0;
   virtual char *GetFile(const char *lfn, const char *localdest = 0) = 0;
   virtual Grid_ResultHandle_t GetPhysicalFileNames(const char *lfn) = 0;

   //--- file catalog queries
   virtual Grid_ResultHandle_t Find(const char *path, const char *file,
                                    const char *conditions = 0) = 0;
   virtual Grid_ResultHandle_t FindEx(const char *path, const char *file,
                                      const char *conditions = 0) = 0;

   //--- file catalog meta data management
   virtual Int_t AddTag(const char *ldn, const char *tagName) = 0;
   virtual Int_t RemoveTag(const char *ldn, const char *tagName) = 0;
   virtual Grid_ResultHandle_t GetTags(const char *ldn) = 0;
   virtual Int_t AddAttributes(const char *lfn, const char *tagName,
                               Int_t inargs, ...) = 0;
   virtual Int_t AddAttribute(const char *lfn, const char *tagName,
                              const char *attrname, const char *attrval) = 0;
   virtual Int_t DeleteAttribute(const char *lfn, const char *tagName,
                                 const char *attrname) = 0;
   virtual Grid_ResultHandle_t GetAttributes(const char *lfn,
                                             const char *tagName) = 0;

   //--- job management
   virtual Grid_JobId_t      SubmitJob(const char *jdlFile) = 0;
   virtual Grid_JobStatus_t *GetJobStatus(Grid_JobId_t jobId) = 0;
   virtual Int_t             KillJob(Grid_JobId_t jobId) = 0;
   virtual Grid_JobId_t      ResubmitJob(Grid_JobId_t jobId) = 0;

   //--- file access
   virtual Grid_AccessPath_t *GetAccessPath(const char *lfn, Bool_t mode = kFALSE,
                                            const char *msn = 0) = 0;
   virtual char *GetFileUrl(const char *msn, const char *path) = 0;

   //--- file access posix interface
   virtual Grid_FileHandle_t GridOpen(const char *lfn, Int_t flags,
                                      UInt_t mode = 0) = 0;

   virtual Int_t GridClose(Grid_FileHandle_t handle) = 0;
   virtual Int_t GridRead(Grid_FileHandle_t handle, void *buffer,
                          Long_t size, Long64_t offset) = 0;
   virtual Int_t GridWrite(Grid_FileHandle_t handle, void *buffer,
                           Long_t size, Long64_t offset) = 0;
   virtual Int_t GridFstat(Grid_FileHandle_t handle,
                           gridstat_t *statbuf) = 0;
   virtual Int_t GridFsync(Grid_FileHandle_t handle) = 0;
   virtual Int_t GridFchmod(Grid_FileHandle_t handle, UInt_t mode) = 0;
   virtual Int_t GridFchown(Grid_FileHandle_t handle, UInt_t owner,
                            UInt_t group) = 0;
   virtual Int_t GridLink(const char *source, const char *target) = 0;
   virtual Int_t GridSymlink(const char *source, const char *target) = 0;
   virtual Int_t GridReadlink(const char *lfn, char *buf, size_t bufsize) = 0;

   virtual Int_t GridStat(const char *lfn, gridstat_t *statbuf) = 0;
   virtual Int_t GridLstat(const char *lfn, gridstat_t *statbuf) = 0;

   virtual Grid_FileHandle_t GridOpendir(const char *dir) = 0;
   virtual const Grid_FileEntry_t *GridReaddir(Grid_FileHandle_t handle) = 0;
   virtual Int_t GridClosedir(Grid_FileHandle_t handle) = 0;

   //--- PROOF interface
   virtual void SetGridProof(TGridProof *proof) { fProof = proof; }
   virtual TGridProof *GetGridProof() const { return fProof; }
   virtual Grid_ProofSession_t *RequestProofSession(const char *user,
                                                    Int_t nsites,
                                                    void **sites,
                                                    void **ntimes,
                                                    time_t starttime,
                                                    time_t duration) = 0;
   virtual Grid_ProofSessionStatus_t GetProofSessionStatus(Grid_ProofSession_t *proofSession) = 0;
   virtual void   ListProofDaemons() = 0;
   virtual void   ListProofSessions(Int_t sessionId = 0) = 0;
   virtual Bool_t KillProofSession(Int_t sessionId) = 0;
   virtual Bool_t KillProofSession(Grid_ProofSession_t *proofSession) = 0;

   //--- plugin factory
   virtual TGridResult *CreateGridResult(Grid_ResultHandle_t handle) = 0;
   virtual TGridProof  *CreateGridProof() = 0;

   //--- load desired plugin and setup conection to GRID
   static TGrid *Connect(const char *grid, const char *uid = 0,
                         const char *pw = 0, const char *options = 0);

   ClassDef(TGrid,0)  // ABC defining interface to GRID services
};

R__EXTERN TGrid *gGrid;

#endif
