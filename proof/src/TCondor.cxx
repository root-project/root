// @(#)root/proof:$Name:  $:$Id: TCondor.cxx,v 1.5 2003/09/23 08:54:50 rdm Exp $
// Author: Maarten Ballintijn   06/12/03

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCondor                                                              //
//                                                                      //
// Interface to the Condor system. TCondor provides a (partial) API for //
// querying and controlling the Condor system, including experimental   //
// extensions like COD (computing on demand)                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCondor.h"
#include "TList.h"
#include "TSystem.h"
#include "TObjString.h"
#include "TRegexp.h"
#include "TProofDebug.h"
#include "Riostream.h"


ClassImp(TCondorSlave)
ClassImp(TCondor)


//______________________________________________________________________________
TCondor::TCondor(const char *pool) : fPool(pool), fState(kFree)
{
   // Create Condor interface object. Uses Condor apps since there is no
   // API yet.

   fClaims = new TList;

   // hack our path :-/
   TString path = gSystem->Getenv("PATH");
   path = "/opt/condor/bin:" + path;
   gSystem->Setenv("PATH",path);
   gSystem->Setenv("CONDOR_CONFIG","/opt/condor/etc/condor_config");
}


//______________________________________________________________________________
TCondor::~TCondor()
{
   // Cleanup Condor interface.

   PDB(kCondor,1) Info("~TCondor","fState %d", fState );

   if (fState != kFree) {
      Release();
   }
   delete fClaims;
}


//______________________________________________________________________________
void TCondor::Print(Option_t * opt) const
{
   cout << "OBJ: " << IsA()->GetName()
      << "\tPool: \"" << fPool << "\""
      << "\tState: " << fState << endl;
   fClaims->Print(opt);
}


//______________________________________________________________________________
TCondorSlave *TCondor::ClaimVM(const char *vm, const char * /*cmd*/, Int_t &port)
{
   // Claim a VirtualMachine for PROOF usage.

   TString claimCmd = Form("condor_cod request -name %s -timeout 10 2>/dev/null", vm );

   PDB(kCondor,2) Info("ClaimVM","command: %s", claimCmd.Data());
   FILE  *pipe = gSystem->OpenPipe(claimCmd, "r");

   if (!pipe) {
      SysError("ClaimVM","cannot run command: %s", claimCmd.Data());
      return 0;
   }

   TString claimId;
   TString line;
   while (line.Gets(pipe)) {
      PDB(kCondor,3) Info("ClaimVM","line = %s", line.Data());

      if (line.BeginsWith("ClaimId = \"")) {
         line.Remove(0, line.Index("\"")+1);
         line.Chop(); // remove trailing "
         claimId = line;
         PDB(kCondor,1) Info("ClaimVM","claim = '%s'", claimId.Data());
         TRegexp r("[0-9]*$");
         TString num = line(r);
         port = 37000 + atoi(num.Data());
         PDB(kCondor,1) Info("ClaimVM","port = %d", port);
      }
   }

   Int_t r = gSystem->ClosePipe(pipe);
   PDB(kCondor,1) Info("ClaimVM","command: %s returned %d", claimCmd.Data(), r);

   TString jobad("jobad");
   FILE *jf = gSystem->TempFileName(jobad);

   if (jf == 0) return 0;

   fputs("JobUniverse = 5\n", jf); // vanilla
   fputs("Cmd = \"/usr/local/root/bin/proofd\"\n", jf);
   fputs("Iwd = \"/tmp\"\n", jf);
   fputs("In = \"/dev/null\"\n", jf);
   fprintf(jf, "Out = \"/tmp/proofd.out.%d\"\n", port);
   fprintf(jf, "Err = \"/tmp/proofd.err.%d\"\n", port);
   fprintf(jf, "Args = \"-f -p %d -d 3 /usr/local/root\"\n", port);
   fclose(jf);

   TString activateCmd = Form("condor_cod activate -id '%s' -jobad %s",
                              claimId.Data(), jobad.Data() );

   PDB(kCondor,2) Info("ClaimVM","command: %s", activateCmd.Data());
   pipe = gSystem->OpenPipe(activateCmd, "r");

   if (!pipe) {
      SysError("ClaimVM","cannot run command: %s", activateCmd.Data());
      return 0;
   }

   while (line.Gets(pipe)) {
      PDB(kCondor,3) Info("ClaimVM","Activate: line = %s", line.Data());
   }

   r = gSystem->ClosePipe(pipe);
   PDB(kCondor,1) Info("ClaimVM","command: %s returned %d", activateCmd.Data(), r);

   gSystem->Unlink(jobad.Data());

   // TODO: get info at the start for all nodes ...
   TCondorSlave *claim = new TCondorSlave;
   claim->fClaimID = claimId;
   TString node(vm);
   node = node.Remove(0, node.Index("@")+1);
   claim->fHostname = node;
   claim->fPort = port;
   if ( !GetVmInfo(vm, claim->fImage, claim->fPerfIdx) ) {
      // assume vm is gone
      delete claim;
      return 0;
   }

   return claim;
}


//______________________________________________________________________________
TList *TCondor::GetVirtualMachines() const
{
   // Get the names of the virtual machines in the pool.
   // Return a TList of TObjString or 0 in case of failure

   TString poolopt = fPool ? "" : Form("-pool %s", fPool.Data());
   TString cmd = Form("condor_status %s -format \"%%s\\n\" Name", poolopt.Data());

   PDB(kCondor,2) Info("GetVirtualMachines","command: %s", cmd.Data());

   FILE  *pipe = gSystem->OpenPipe(cmd, "r");

   if (!pipe) {
      SysError("GetVirtualMachines","cannot run command: %s", cmd.Data());
      return 0;
   }

   TString line;
   TList *l = new TList;
   while (line.Gets(pipe)) {
      PDB(kCondor,3) Info("GetVirtualMachines","line = %s", line.Data());
      if (line != "") l->Add(new TObjString(line));
   }

   Int_t r = gSystem->ClosePipe(pipe);
   PDB(kCondor,1) Info("GetVirtualMachines","command: %s returned %d", cmd.Data(), r);

   return l;
}


//______________________________________________________________________________
TList *TCondor::Claim(Int_t n, const char *cmd)
{
   if (fState != kFree) {
      Error("Claim","not in state Free");
      return 0;
   }

   TList *vms = GetVirtualMachines();

   TIter next(vms);
   TObjString *vm;
   for(Int_t i=0; i < n && (vm = (TObjString*) next()) != 0; i++ ) {
      Int_t port = 17000+i; // hard code port for the moment
      TCondorSlave *claim = ClaimVM(vm->GetName(), cmd, port);
      if (claim != 0) {

         fClaims->Add(claim);

      }
   }

   fState = kActive;

   return fClaims;
}


//______________________________________________________________________________
Bool_t TCondor::SetState(EState state)
{
   TIter next(fClaims);
   TCondorSlave *claim;
   while((claim = (TCondorSlave*) next()) != 0) {
      TString cmd = Form("condor_cod %s -id '%s'",
                         state == kSuspended ? "suspend" : "resume",
                         claim->fClaimID.Data());

      PDB(kCondor,2) Info("SetState","command: %s", cmd.Data());
      FILE  *pipe = gSystem->OpenPipe(cmd, "r");

      if (!pipe) {
         SysError("SetState","cannot run command: %s", cmd.Data());
         return kFALSE;
      }

      TString line;
      while (line.Gets(pipe)) {
         PDB(kCondor,3) Info("SetState","line = %s", line.Data());
      }

      Int_t r = gSystem->ClosePipe(pipe);
      PDB(kCondor,1) Info("SetState","command: %s returned %d", cmd.Data(), r);
   }

   fState = state;
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TCondor::Suspend()
{
   if (fState != kActive) {
      Error("Suspend","not in state Active");
      return kFALSE;
   }

   return SetState(kSuspended);
}


//______________________________________________________________________________
Bool_t TCondor::Resume()
{
   if (fState != kSuspended) {
      Error("Suspend","not in state Suspended");
      return kFALSE;
   }

   return SetState(kActive);
}


//______________________________________________________________________________
Bool_t TCondor::Release()
{
   if (fState == kFree) {
      Error("Suspend","not in state Active or Suspended");
      return kFALSE;
   }

   TCondorSlave *claim;
   while((claim = (TCondorSlave*) fClaims->First()) != 0) {
      TString cmd = Form("condor_cod release -id '%s'", claim->fClaimID.Data());

      PDB(kCondor,2) Info("SetState","command: %s", cmd.Data());
      FILE  *pipe = gSystem->OpenPipe(cmd, "r");

      if (!pipe) {
         SysError("Release","cannot run command: %s", cmd.Data());
         return kFALSE;
      }

      TString line;
      while (line.Gets(pipe)) {
         PDB(kCondor,3) Info("Release","line = %s", line.Data());
      }

      Int_t r = gSystem->ClosePipe(pipe);
      PDB(kCondor,1) Info("Release","command: %s returned %d", cmd.Data(), r);

      fClaims->Remove(claim);
      delete claim;
   }

   fState = kFree;
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TCondor::GetVmInfo(const char *vm, TString &image, Int_t &perfidx) const
{
   TString cmd = Form("condor_status -format \"%%d:\" Mips -format \"%%s\\n\" FileSystemDomain "
                      "-const 'Name==\"%s\"'", vm);

   PDB(kCondor,2) Info("GetVmInfo","command: %s", cmd.Data());
   FILE  *pipe = gSystem->OpenPipe(cmd, "r");

   if (!pipe) {
      SysError("GetVmInfo","cannot run command: %s", cmd.Data());
      return 0;
   }

   TString line;
   while (line.Gets(pipe)) {
      PDB(kCondor,3) Info("GetVmInfo","line = %s", line.Data());
      if (line != "") {
         TString amips = line(TRegexp("^[0-9]*"));
         perfidx = atoi(amips);
         image = line(TRegexp("[^:]+$"));
         break;
      }
   }

   Int_t r = gSystem->ClosePipe(pipe);
   PDB(kCondor,1) Info("GetVmInfo","command: %s returned %d", cmd.Data(), r);

   return kTRUE;
}


//______________________________________________________________________________
TString TCondor::GetImage(const char *host) const
{
   TString cmd = Form("condor_status -direct %s -format \"Image:%%s\\n\" "
                      "FileSystemDomain", host);

   PDB(kCondor,2) Info("GetImage","command: %s", cmd.Data());

   FILE  *pipe = gSystem->OpenPipe(cmd, "r");

   if (!pipe) {
      SysError("GetImage","cannot run command: %s", cmd.Data());
      return 0;
   }

   TString image;
   TString line;
   while (line.Gets(pipe)) {
      PDB(kCondor,3) Info("GetImage","line = %s", line.Data());
      if (line != "") {
         image = line(TRegexp("[^:]+$"));
         break;
      }
   }

   Int_t r = gSystem->ClosePipe(pipe);
   PDB(kCondor,1) Info("GetImage","command: %s returned %d", cmd.Data(), r);

   return image;
}


//______________________________________________________________________________
void TCondorSlave::Print(Option_t * /*opt*/ ) const
{
   cout << "OBJ: " << IsA()->GetName()
      << " " << fHostname << ":" << fPort
      << "  Perf: " << fPerfIdx
      << "  Image: " << fImage << endl;
}

