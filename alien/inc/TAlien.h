// @(#)root/alien:$Name:  $:$Id: TAlien.h,v 1.6 2002/05/31 11:29:23 rdm Exp $
// Author: Fons Rademakers   13/5/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
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
// Class defining interface to AliEn GRID services.                     //
//                                                                      //
// To start a local API Grid service, use                               //
//   - TGrid::Connect("alien://localhost");                             //
//   - TGrid::Connect("alien://");                                      //
//                                                                      //
// To force to connect to a running API Service, use                    //
//   - TGrid::Connect("alien://<apihosturl>/?direct");                  //
//                                                                      //
// To get a remote API Service from the API factory service, use        //
//   - TGrid::Connect("alien://<apifactoryurl>");                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGrid
#include "TGrid.h"
#endif

#ifndef ROOT_TAlienResult
#include "TAlienResult.h"
#endif

#ifndef ROOT_TAlienProof
#include "TAlienProof.h"
#endif

class TAliEnAPI;


class TAlien : public TGrid {

private:
   TString    fGridUrl;         // specifies the GRID URL
   TString    fUid;             // specifies the User identifier (user name)
   TString    fPw;              // specifies a user password;
   TString    fOptions;         // specifies options for the Grid
   TAliEnAPI *fAPI;             // pointer to AliEn API driver

   Int_t API(const char *apiServerUrl = 0, const char *user = 0);

public:
   TAlien(const char *gridurl, const char *uid = 0, const char *pw = 0,
          const char *options = 0);
   virtual ~TAlien();

   const char *GetUser();
   Bool_t      IsConnected() const { return fAPI ? kTRUE : kFALSE; }

   //--- file catalog browsing
   Grid_ResultHandle_t OpenDir(const char *ldn);

   Grid_Result_t *ReadResult(Grid_ResultHandle_t hResult);
   void           CloseResult(Grid_ResultHandle_t hResult);
   void           ResetResult(Grid_ResultHandle_t hResult);

   //--- file catalog management
   Int_t Mkdir(const char *ldn, Bool_t recursive = kFALSE);
   Int_t Rmdir(const char *dir, Bool_t recursive = kFALSE);
   Int_t Rm(const char *lfn, Bool_t recursive = kFALSE);
   Int_t Cp(const char *sourcelfn, const char *targetlfn);
   Int_t Mv(const char *sourcelfn, const char *targetlfn);
   Int_t Chmod(const char *lfn, mode_t mode);
   Int_t Chown(const char *lfn, const char *owner, const char *group);

   Int_t AddFile(const char *newlfn, const char *pfn, Int_t size =
                 -1, const char *msn = 0, char *guid = 0);
   Int_t AddFileMirror(const char *lfn, const char *pfn, const char *msn);
   Int_t RegisterFile(const char *lfn, const char *pfn, const char *msn = "");
   char *GetFile(const char *lfn, const char *localdest = 0);
   Grid_ResultHandle_t GetPhysicalFileNames(const char *lfn);

   //--- file catalog queries
   Grid_ResultHandle_t Find(const char *path, const char *file,
                            const char *conditions = 0);
   Grid_ResultHandle_t FindEx(const char *path, const char *file,
                              const char *conditions = 0);

   //--- file catalog meta data management
   Int_t AddTag(const char *ldn, const char *tagName);
   Int_t RemoveTag(const char *ldn, const char *tagName);
   Grid_ResultHandle_t GetTags(const char *ldn);
   Int_t AddAttributes(const char *lfn, const char *tagName,
                       Int_t inargs, ...);
   Int_t AddAttribute(const char *lfn, const char *tagName,
                      const char *attrname, const char *attrval);
   Int_t DeleteAttribute(const char *lfn, const char *tagName,
                         const char *attrname);
   Grid_ResultHandle_t GetAttributes(const char *lfn, const char *tagName);

   //--- job management
   Grid_JobId_t      SubmitJob(const char *jdlFile);
   Grid_JobStatus_t *GetJobStatus(Grid_JobId_t jobId);
   Int_t             KillJob(Grid_JobId_t jobId);
   Grid_JobId_t      ResubmitJob(Grid_JobId_t jobId);

   //--- file access
   Grid_AccessPath_t *GetAccessPath(const char *lfn, Bool_t mode = kFALSE,
                                    const char *msn = 0);
   char *GetFileUrl(const char *msn, const char *path);

   //--- file access posix interface
   Grid_FileHandle_t GridOpen(const char *lfn, Int_t flags, mode_t mode);
   Int_t GridClose(Grid_FileHandle_t handle);
   Int_t GridRead(Grid_FileHandle_t handle, void *buffer, Long_t size,
                  Long64_t offset);
   Int_t GridWrite(Grid_FileHandle_t handle, void *buffer, Long_t size,
                   Long64_t offset);

   Int_t GridFstat(Grid_FileHandle_t handle, gridstat_t *statbuf);

   Int_t GridFsync(Grid_FileHandle_t handle);
   Int_t GridFchmod(Grid_FileHandle_t handle, mode_t mode);
   Int_t GridFchown(Grid_FileHandle_t handle, uid_t owner, gid_t group);
   Int_t GridLink(const char *source, const char *target);
   Int_t GridSymlink(const char *source, const char *target);
   Int_t GridReadlink(const char *lfn, char *buf, size_t bufsize);

   Int_t GridStat(const char *lfn, gridstat_t *statbuf);
   Int_t GridLstat(const char *lfn, gridstat_t *statbuf);

   Grid_FileHandle_t GridOpendir(const char *dir);
   const Grid_FileEntry_t *GridReaddir(Grid_FileHandle_t handle);
   Int_t GridClosedir(Grid_FileHandle_t handle);

   //--- PROOF interface
   Grid_ProofSession_t *RequestProofSession(const char *user, Int_t nsites,
                                            void **sites, void **ntimes,
                                            time_t starttime,
                                            time_t duration);
   Grid_ProofSessionStatus_t GetProofSessionStatus(Grid_ProofSession_t *proofSession);
   void   ListProofDaemons();
   void   ListProofSessions(Int_t sessionId = 0);
   Bool_t KillProofSession(Int_t sessionId);
   Bool_t KillProofSession(Grid_ProofSession_t *proofSession);

   //--- plugin factory
   TGridResult *CreateGridResult(Grid_ResultHandle_t handle)
   {
      return new TAlienResult(handle);
   }
   TGridProof *CreateGridProof()
   {
      SetGridProof(new TAlienProof());
      return GetGridProof();
   }

   ClassDef(TAlien,0)  // Interface to AliEn GRID services
};

#endif
