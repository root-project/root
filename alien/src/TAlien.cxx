// @(#)root/alien:$Name:  $:$Id: TAlien.cxx,v 1.9 2003/11/13 15:15:11 rdm Exp $
// Author: Andreas Peters     04.09.2003

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TUrl.h"
#include "TAlien.h"
#include <stdlib.h>

#define ALIEN_POSIX
#include <AliEnAPI++.h>

using namespace std;


ClassImp(TAlien)

//______________________________________________________________________________
TAlien::TAlien(const char *gridurl, const char *uid, const char *pw,
               const char *options)
{
   // Create a connection to AliEn via its API Service.

   fAPI     = 0;
   fGridUrl = gridurl;
   fUid     = uid;
   fPw      = pw;
   fOptions = options;
   gGrid    = this;
   if ((API(fGridUrl)) == 0) {
      fAPI->PrintMOTD();
      TUrl gridUrl(gridurl);
      fGrid = gridUrl.GetProtocol();
   } else {
      Error("TAlien", "could not connect to the AliEn API Service");
      MakeZombie();
   }
}

//______________________________________________________________________________
TAlien::~TAlien()
{
   // Close connection to the AliEn API Service.

   if (fAPI)
      fAPI->StopApiServer();

   delete fAPI;
   fAPI = 0;
}

//______________________________________________________________________________
Int_t TAlien::API(const char *apiserverurl, const char *user)
{
   // Connect to AliEn API Serivce. Returns -1 in case of error.

   TUrl apiUrl(apiserverurl);

   fAPI = new TAliEnAPI();

   if (!fAPI)
      return -1;

   if ((strlen(apiUrl.GetHost()) == 0)
       || (!(strcmp(apiUrl.GetHost(), "localhost")))) {
      // try to reuse the old API
      if (!fAPI->ReuseApiServer(user)) {
         return 0;
      }
      // start an API service locally
      if (fAPI->StartApiServer(user)) {
         Error("TAlien", "error starting the API Server");
         return -1;
      }
   } else {
      // if options = "direct", we set the Api URL by hand
      if (!(strcmp(apiUrl.GetOptions(), "direct"))) {
         // set to another API service
         TString newUrl = "http://";
         newUrl += apiUrl.GetHost();
         newUrl += ":";
         newUrl += apiUrl.GetPort();

         fAPI->SetApiServerUrl(newUrl);
      } else {
         // call the AliEn Api factory service
         return fAPI->ApiFactory(apiUrl.GetUrl());
      }
   }
   return 0;
}

//______________________________________________________________________________
const char *TAlien::GetUser()
{
   // Get the user name of the API service. Return 0 in case not connected
   // to API Service. Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->GetApiUser();
}

//______________________________________________________________________________
Grid_ProofSession_t *TAlien::RequestProofSession(const char *user,
                                                 Int_t nsites,
                                                 void **sites,
                                                 void **ntimes,
                                                 time_t starttime,
                                                 time_t duration)
{
   // Request a PROOF session at <starttime> for <duration> seconds and
   // user <user>. Returns 0 in case not connected to API Service.
   // Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return (Grid_ProofSession_t *) fAPI->RequestPROOFSession(user, nsites,
           (string **) sites, (string **) ntimes, starttime, duration);
}

//______________________________________________________________________________
Grid_ProofSessionStatus_t TAlien::GetProofSessionStatus(Grid_ProofSession_t *proofSession)
{
   // Get the status of a PROOF session. Return -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->QueryPROOFSession((TAliEnAPI::ProofSession_T *) proofSession);
}

//______________________________________________________________________________
void TAlien::ListProofDaemons()
{
   // List PROOF daemons.

   if (!fAPI)
      return;

   return fAPI->ListPROOFDaemons();
}

//______________________________________________________________________________
void TAlien::ListProofSessions(Int_t sessionid)
{
   // List proof sessions for all or <sessionid>.

   if (!fAPI)
      return;

   return fAPI->ListPROOFSessions(sessionid);
}

//______________________________________________________________________________
Bool_t TAlien::KillProofSession(Int_t sessionid)
{
   // Kill PROOF session with ID <sessionid>. Returns false in case of error.

   if (!fAPI)
      return kFALSE;

   return fAPI->CancelPROOFSession(sessionid);
}

//______________________________________________________________________________
Bool_t TAlien::KillProofSession(Grid_ProofSession_t * proofSession)
{
   // Kill PROOF session <proofSession>. Returns false in case of error.

   if (!fAPI)
      return kFALSE;

   return fAPI->CancelPROOFSession((TAliEnAPI::ProofSession_T *) proofSession);
}

//______________________________________________________________________________
Grid_ResultHandle_t TAlien::OpenDir(const char *ldn)
{
   // Open a catalog directory pointed by logical directory name <ldn>
   // (like posix opendir). Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->OpenDir(ldn);
}

//______________________________________________________________________________
Grid_Result_t *TAlien::ReadResult(Grid_ResultHandle_t hResult)
{
   // Fetch a result from result handle <hResult>. Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return (Grid_Result_t *) fAPI->ReadResult(hResult);
}

//______________________________________________________________________________
void TAlien::CloseResult(Grid_ResultHandle_t hResult)
{
   // Close a result list with handle <hResult>.

   if (!fAPI)
      return;

   return fAPI->CloseResult(hResult);
}

//______________________________________________________________________________
void TAlien::ResetResult(Grid_ResultHandle_t hResult)
{
   // Reset a result list with handle <hResult>.

   if (!fAPI)
      return;

   return fAPI->ResetResult(hResult);
}

//______________________________________________________________________________
Int_t TAlien::Mkdir(const char *ldn, Bool_t recursive)
{
   // Make a directory pointed by logical directory name <ldn> in the file
   // catalog. If <recursive> is true, all directories in the path, which
   // do not exist, are created. Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->MkDir(ldn, recursive);
}

//______________________________________________________________________________
Int_t TAlien::Rmdir(const char *ldn, Bool_t recursive)
{
   // Remove a directory pointed by logical directory name <ldn> in the file
   // catalog. If <recursive> is true, all contained subdirectories and files
   // in <ldn> are also removed. Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->RmDir(ldn, recursive);
}

//______________________________________________________________________________
Int_t TAlien::Rm(const char *lfn, Bool_t recursive)
{
   // Removes a file from AliEn. If <recursive> is true and <lfn> contains
   // a wildcard, also subdirectories are removed recursively.
   // Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->Rm(lfn, recursive);
}

//______________________________________________________________________________
Int_t TAlien::Cp(const char *sourcelfn, const char *targetlfn)
{
   // Copies <sourcelfn> to <targetlfn> in the file catalog.
   // Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->Cp(sourcelfn, targetlfn);
}

//______________________________________________________________________________
int TAlien::Mv(const char *sourcelfn, const char *targetlfn)
{
   // Moves <sourcelfn> to <targetlfn> in the file catalog.
   // Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->Mv(sourcelfn, targetlfn);
}

//______________________________________________________________________________
Int_t TAlien::Chmod(const char *lfn, UInt_t mode)
{
   // Changes the permission of <lfn> to <mode>.
   // Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->Chmod(lfn, mode);
}

//______________________________________________________________________________
Int_t TAlien::Chown(const char *lfn, const char *owner, const char *group)
{
   // Changes the owner and group of <lfn> to <owner> and <group>.
   // Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->Chown(lfn, owner, group);
}

//______________________________________________________________________________
Int_t TAlien::AddFile(const char *newlfn, const char *pfn, Int_t size,
                      const char *msn, char *guid)
{
   // Adds an LFN entry to the file catalog:
   // - <pfn>  = the access URL
   // - <size> = file size in byte
   // - <msn>  = mass storage name
   // - <guid> = Global Identifier
   // Returns -1 on error.
   // If size=-1, AliEn will try to guess the size from <pfn>.

   if (!fAPI)
      return -1;

   return fAPI->AddFile(newlfn, pfn, size, msn, guid);
}

//______________________________________________________________________________
Int_t TAlien::AddFileMirror(const char *lfn, const char *pfn,
                            const char *msn)
{
   // Add a mirror <pfn> to <lfn>, which resides in the mass storage
   // system <msn>. Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->AddFileMirror(lfn, pfn, msn);
}

//______________________________________________________________________________
Int_t TAlien::RegisterFile(const char *lfn, const char *pfn, const char *msn)
{
   // Register the file located under <pfn> as <lfn> in the mass storage
   // system <msn>. If <msn> is omitted, the closest mass storage system is
   // chosen. WARNING! This function makes only sense, if the API server runs
   // on the local machine. Returns -1 in case of error.

   if (!fAPI)
      return -1;

   TString pfnmsn = pfn;
   pfnmsn += " ";
   pfnmsn += msn;

   return fAPI->RegisterFile(lfn, pfnmsn.Data());
}

//______________________________________________________________________________
char *TAlien::GetFile(const char *lfn, const char *localdest)
{
   // Copy the file <lfn> from the file catalog to the local file <localdest>.
   // If <localdest> is omitted, the file is copied somewhere to a temporary
   // place. WARNING! This function makes only sense, if the API server runs
   // on the local machine. Returns 0 in case of error. If localdest is
   // not 0, file is copied to localdest.

   if (!fAPI)
      return 0;

   return fAPI->GetFile(lfn, localdest);
}

//______________________________________________________________________________
Grid_ResultHandle_t TAlien::GetPhysicalFileNames(const char *lfn)
{
   // Get a list of physical file names for <lfn>. The results can be read
   // with the TGridResult class or with 'ReadResult'/'CloseResult'/
   // 'ResetResult. Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->GetPhysicalFileNames(lfn);
}

//______________________________________________________________________________
Grid_ResultHandle_t TAlien::Find(const char *path, const char *file,
                                 const char *conditions)
{
   // Finds all files like <file> starting from <path>, which fullfill
   // <conditions>:
   // - path = catalog path to start searching
   // - file = filename or wildcard f.e. test.root, *.root test*.root ab?.root
   // - conditions = "Tag1:Attr1='Value1' and/or Tag2:Attr2='Value2"
   // Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->Find(path, file, conditions);
}

//______________________________________________________________________________
Grid_ResultHandle_t TAlien::FindEx(const char *path, const char *file,
                                   const char *conditions)
{
   // Finds all files like <file> starting from <path>, which fullfill
   // <conditions>:
   // - path = catalog path to start searching
   // - file = filename or wildcard f.e. test.root, *.root test*.root ab?.root
   // - conditions = "Tag1:Attr1='Value1' and/or Tag2:Attr2='Value2"
   // It returns additionally the list of <pfn> and <msn> for each <lfn>.
   // Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->FindEx(path, file, conditions);
}

//______________________________________________________________________________
Int_t TAlien::AddTag(const char *ldn, const char *tagName)
{
   // Add a tag name <tagName> to a logical directory name <ldn> in the file
   // catalog. Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->AddTag(ldn, tagName);
}

//______________________________________________________________________________
Int_t TAlien::RemoveTag(const char *ldn, const char *tagName)
{
   // Remove a tag name <tagName> from a logical directory name <ldn> in the
   // file catalog. Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->RemoveTag(ldn, tagName);
}

//______________________________________________________________________________
Grid_ResultHandle_t TAlien::GetTags(const char *ldn)
{
   // Gets a list of tags for the logical directory name <ldn> from the file
   // catalog. Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->GetTags(ldn);
}

//______________________________________________________________________________
Int_t TAlien::AddAttributes(const char *lfn, const char *tagName,
                            Int_t inargs, ...)
{
   // Add several attributes (inargs/2) <"name","value","name","value"....>
   // to the tag <tagName> to <lfn>. Returns -1 in case of error.

   if (!fAPI)
      return -1;

   va_list ap;
   va_start(ap, inargs);

   int result = fAPI->VAddAttributes(lfn, tagName, inargs, ap);
   va_end(ap);
   return result;
}

//______________________________________________________________________________
Int_t TAlien::AddAttribute(const char *lfn, const char *tagName,
                           const char *attrname, const char *attrval)
{
   // Add an attribute <attrname> with value <attrval> to the tag <tagName>
   // to <lfn>. Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->AddAttribute(lfn, tagName, attrname, attrval);
}

//______________________________________________________________________________
Int_t TAlien::DeleteAttribute(const char *lfn, const char *tagName,
                             const char *attrname)
{
   // Delete an attribute <attrname> from tag <tagName> of <lfn>.
   // Returns -1 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->DeleteAttribute(lfn, tagName, attrname);
}

//______________________________________________________________________________
Grid_ResultHandle_t TAlien::GetAttributes(const char *lfn, const char *tagName)
{
   // Get a list of attributes for tag <tagName> for <lfn>.
   // Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->GetAttributes(lfn, tagName);
}

//______________________________________________________________________________
Grid_JobId_t TAlien::SubmitJob(const char *jdlFile)
{
   // Submit a job with JDL file <jdlFile> to AliEN.
   // The <jdlFile> should be an lfn. If the API server run's locally,
   // one can specify reading a local file "file.jdl" by specifying
   // '<file.jdl>' as <jdlFile>
   // Returns a positive job ID or a negative error value.

   if (!fAPI)
      return -1;

   return fAPI->SubmitJob(jdlFile);
}

//______________________________________________________________________________
Grid_JobStatus_t *TAlien::GetJobStatus(Grid_JobId_t jobId)
{
   // Returns the status of a job with ID <jobId> from the grid queue,
   // or 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->GetJobStatus(jobId);
}

//______________________________________________________________________________
Int_t TAlien::KillJob(Grid_JobId_t jobId)
{
   // Kill a job in the grid queue with ID <jobId>.
   // Returns 0 for success or a negative error value.

   if (!fAPI)
      return -1;

   return fAPI->KillJob(jobId);
}

//______________________________________________________________________________
Grid_JobId_t TAlien::ResubmitJob(Grid_JobId_t jobId)
{
   // Resubmits the job with job ID <jobId>.
   // Returns the job ID of the new job.

   if (!fAPI)
      return -1;

   return fAPI->ResubmitJob(jobId);
}

//______________________________________________________________________________
Grid_AccessPath_t *TAlien::GetAccessPath(const char *lfn, Bool_t write,
                                         const char *msn)
{
   // Returns an URL for a <lfn> access using AliEn I/O daemons,
   // 'write = false' means read access
   // 'write = true'  means write access
   // The <msn> can be specified for access of <lfn> from a specific mass
   // storage system. Returns 0 in case of error.
   // The result has to be freed by the user!

   if (!fAPI)
      return 0;

   return fAPI->GetAccessPath(lfn, write, msn);
}

//______________________________________________________________________________
char *TAlien::GetFileUrl(const char *msn, const char *path)
{
   // Builds an URL by specifying the physical path at the MSN and the MSN name
   // Returns 0 in case of error. Result has to be freed by the user.

   if (!fAPI)
      return 0;

   return fAPI->GetFileURL(msn, path);
}

//______________________________________________________________________________
Grid_FileHandle_t TAlien::GridOpen(const char *lfn, Int_t flags, UInt_t mode)
{
   // POSIX open for grid files. Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->open(lfn, flags, mode);
}

//______________________________________________________________________________
Int_t TAlien::GridClose(Grid_FileHandle_t handle)
{
   // POSIX close for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->close(handle);
}

//______________________________________________________________________________
Int_t TAlien::GridRead(Grid_FileHandle_t handle, void *buffer, Long_t size,
                       Long64_t offset)
{
   // POSIX read for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->read(handle, buffer, size, offset);
}

//______________________________________________________________________________
Int_t TAlien::GridWrite(Grid_FileHandle_t handle, void *buffer, Long_t size,
                        Long64_t offset)
{
   // POSIX write for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->write(handle, buffer, size, offset);
}

//______________________________________________________________________________
Int_t TAlien::GridFstat(Grid_FileHandle_t handle, gridstat_t *statbuf)
{
   // POSIX fstat for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->fstat(handle, statbuf);
}

//______________________________________________________________________________
Int_t TAlien::GridFsync(Grid_FileHandle_t handle)
{
   // POSIX fsync for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->fsync(handle);
}

//______________________________________________________________________________
Int_t TAlien::GridFchmod(Grid_FileHandle_t handle, UInt_t mode)
{
   // POSIX fchmod for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->fchmod(handle, mode);
}

//______________________________________________________________________________
Int_t TAlien::GridFchown(Grid_FileHandle_t handle, UInt_t owner, UInt_t group)
{
   // POSIX fchown for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->fchown(handle, owner, group);
}

//______________________________________________________________________________
Int_t TAlien::GridLink(const char *source, const char *target)
{
   // POSIX link for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return -1;
   // return fAPI->link(source, target); // not implemented in AliEn yet
}

//______________________________________________________________________________
Int_t TAlien::GridSymlink(const char *source, const char *target)
{
   // POSIX symlink for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return -1;
   // return fAPI->symlink(source, target); // not implemented in AliEn yet
}

//______________________________________________________________________________
Int_t TAlien::GridReadlink(const char *lfn, char *buf, size_t bufsize)
{
   // POSIX readlink for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return -1;
   // return fAPI->readlink(lfn, buf, bufsize); // not implemented in AliEn yet
}

//______________________________________________________________________________
Int_t TAlien::GridStat(const char *lfn, gridstat_t *statbuf)
{
   // POSIX stat for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->stat(lfn, statbuf);
}

//______________________________________________________________________________
Int_t TAlien::GridLstat(const char *lfn, gridstat_t *statbuf)
{
   // POSIX lstat for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->lstat(lfn, statbuf);
}

//______________________________________________________________________________
Grid_FileHandle_t TAlien::GridOpendir(const char *dir)
{
   // POSIX opendir for grid files. Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return fAPI->opendir(dir);
}

//______________________________________________________________________________
const Grid_FileEntry_t *TAlien::GridReaddir(Grid_FileHandle_t handle)
{
   // POSIX readdir for grid files. Returns 0 in case of error.

   if (!fAPI)
      return 0;

   return (Grid_FileEntry_t*) fAPI->readdir(handle);
}

//______________________________________________________________________________
Int_t TAlien::GridClosedir(Grid_FileHandle_t handle)
{
   // POSIX closedir for grid files. Returns < 0 in case of error.

   if (!fAPI)
      return -1;

   return fAPI->closedir(handle);
}
