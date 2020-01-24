// @(#)root/proof:$Id$
// Author: G. Ganis, Oct 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TPackMgr
\ingroup proofkernel

The PROOF package manager contains tools to  manage packages.
This class has been created to eliminate duplications, and to allow for
standalone usage.

*/

#include "TPackMgr.h"

#include "TError.h"
#include "TFile.h"
#include "TFunction.h"
#include "THashList.h"
#include "TList.h"
#include "TMacro.h"
#include "TMD5.h"
#include "TMethodArg.h"
#include "TMethodCall.h"
#include "TObjString.h"
#include "TParameter.h"
#include "TMap.h"
#include "TProof.h" // for constants such as kRM and kLS.
#include "TROOT.h"
#include "TSystem.h"

ClassImp(TPackMgr);


static void DefaultLogger(const char *msg) { Printf("%s", msg); }

THashList *TPackMgr::fgGlobalPackMgrList = 0; // list of package managers for global packages

////////////////////////////////////////////////////////////////////////////////
/// Create a PROOF package manager

TPackMgr::TPackMgr(const char *dir, const char *key)
         : fLogger(DefaultLogger), fName(key), fDir(dir), fLock(dir), fEnabledPackages(0)
{
   // Work with full names
   if (gSystem->ExpandPathName(fDir))
      Warning("TPackMgr", "problems expanding path '%s'", fDir.Data());
   // The lock file in temp
   TString lockname = TString::Format("%s/packdir-lock-%s",
                      gSystem->TempDirectory(), TString(fDir).ReplaceAll("/","%").Data());
   fLock.SetName(lockname);
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy a TPackMgr instance

TPackMgr::~TPackMgr()
{
   // Destroy the lock file
   if (fEnabledPackages) delete fEnabledPackages;
}

////////////////////////////////////////////////////////////////////////////////
/// Wrapper to notofuer / logger

void TPackMgr::Log(const char *msg)
{
   if (fLogger) {
      if (fPfx.IsNull())
         (*fLogger)(msg);
      else
         (*fLogger)(TString::Format("%s: %s", fPfx.Data(), msg));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Method to build a package.
/// Return -1 on error, 0 otherwise

Int_t TPackMgr::Build(const char *pack, Int_t opt)
{
   Int_t rc = 0;

   TLockPathGuard lp(&fLock);

   if (gDebug > 0)
      Info("Build", "building package %s ...", pack);

   TString ocwd = gSystem->WorkingDirectory();
   TString pdir = TString::Format("%s/%s", fDir.Data(), pack);
   gSystem->ChangeDirectory(pdir);

   // check for BUILD.sh and execute
   if (!gSystem->AccessPathName("PROOF-INF/BUILD.sh")) {
      // Notify the upper level
      Log(TString::Format("building %s ...", pack));

      // Read version from file proofvers.txt, and if current version is
      // not the same do a "BUILD.sh clean"
      Bool_t goodver = kTRUE;
      Bool_t savever = kFALSE;
      TString v, r;
      FILE *f = fopen("PROOF-INF/proofvers.txt", "r");
      if (f) {
         v.Gets(f);
         r.Gets(f);
         fclose(f);
         if (opt == TPackMgr::kCheckROOT && v != gROOT->GetVersion()) goodver = kFALSE;
      }
      if (!f || !goodver) {
         if (!gSystem->AccessPathName(pdir, kWritePermission)) {
            savever = kTRUE;
            Log(TString::Format("%s: version change"
                                   " (current: %s, build: %s): cleaning ... ",
                                   pack, gROOT->GetVersion(), v.Data()));
            // Hard cleanup: go up the dir tree
            gSystem->ChangeDirectory(fDir);
            // remove package directory
            gSystem->Exec(TString::Format("%s %s", kRM, pdir.Data()));
            // find gunzip...
            char *gunzip = gSystem->Which(gSystem->Getenv("PATH"), kGUNZIP,
                                          kExecutePermission);
            if (gunzip) {
               TString par;
               par.Form("%s.par", pdir.Data());
               // untar package
               TString cmd;
               cmd.Form(kUNTAR3, gunzip, par.Data());
               rc = gSystem->Exec(cmd);
               if (rc != 0) {
                  Error("Build", "failure executing: %s", cmd.Data());
               } else {
                  // Store md5 in package/PROOF-INF/md5.txt
                  TMD5 *md5local = TMD5::FileChecksum(par);
                  if (md5local) {
                     TString md5f = pdir + "/PROOF-INF/md5.txt";
                     TMD5::WriteChecksum(md5f, md5local);
                     // Go down to the package directory
                     gSystem->ChangeDirectory(pdir);
                     // Cleanup
                     SafeDelete(md5local);
                  } else {
                     Warning("Build", "failure calculating/saving MD5sum for '%s'", par.Data());
                  }
               }
               delete [] gunzip;
            } else {
               Error("Build", "%s not found", kGUNZIP);
               rc = -1;
            }
         } else {
            Log(TString::Format("%s: ROOT version inconsistency (current: %s, build: %s):"
                                 " directory not writable: cannot re-build!!! ",
                                 pack, gROOT->GetVersion(), v.Data()));
            rc = -1;
         }

         if (rc == 0) {
            // To build the package we execute PROOF-INF/BUILD.sh via a pipe
            // so that we can send back the log in (almost) real-time to the
            // (impatient) client. Note that this operation will block, so
            // the messages from builds on the workers will reach the client
            // shortly after the master ones.
            TString ipath(gSystem->GetIncludePath());
            ipath.ReplaceAll("\"","");
            TString cmd;
            cmd.Form("export ROOTINCLUDEPATH=\"%s\" ; PROOF-INF/BUILD.sh", ipath.Data());
            rc = gSystem->Exec(cmd);
            if (rc != 0) {
               Error("Build", "failure executing: %s", cmd.Data());
            } else {
               // Success: write version file
               if (savever) {
                  f = fopen("PROOF-INF/proofvers.txt", "w");
                  if (f) {
                     fputs(gROOT->GetVersion(), f);
                     fputs(TString::Format("\n%s", gROOT->GetGitCommit()), f);
                     fclose(f);
                  }
               }
            }
         }
      } else {
         // Notify the user
         if (gDebug > 0)
            Info("Build", "no PROOF-INF/BUILD.sh found for package %s", pack);
      }
   }
   // Always return to the initial directory
   gSystem->ChangeDirectory(ocwd);

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Method to load a package taking an option const char *
/// Return -1 on error, 0 otherwise

Int_t TPackMgr::Load(const char *pack, const char *opts)
{
   TList *optls = new TList;
   optls->Add(new TObjString(opts));
   Int_t rc = Load(pack, optls);
   optls->SetOwner();
   delete optls;
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Method to load a package taking an option list
/// Return -1 on error, 0 otherwise

Int_t TPackMgr::Load(const char *pack, TList *optls)
{
   Int_t rc = 0;
   TString emsg;

   // If already loaded don't do it again
   if (fEnabledPackages && fEnabledPackages->FindObject(pack)) {
      Log(TString::Format("error: TPackMgr::Load: package %s already loaded", pack));
      return 0;
   }

   // Which pack mgr has the package?
   if (!Has(pack)) {
      // Check the global packages
      TPackMgr *packmgr = 0;
      if (!(packmgr = TPackMgr::GetPackMgr(pack, nullptr))) {
         // Package not found
         Log(TString::Format("error: TPackMgr::Load: failure locating %s ...", pack));
         return -1;
      }
      // Load from there
      return packmgr->Load(pack, optls);
   }

   // We have the package
   TString pdir = TString::Format("%s/%s", fDir.Data(), pack);

   // Check dependencies
   TString deps = TString::Format("%s/PROOF-INF/depends", pdir.Data());
   if (!gSystem->AccessPathName(deps)) {
      TMacro mdeps("deps");
      if (mdeps.ReadFile(deps) > 0) {
         Log(TString::Format("info: TPackMgr::Load: checking dependencies for package %s ...", pack));
         TIter nxl(mdeps.GetListOfLines());
         TObjString *os = 0;
         while ((os = (TObjString *)nxl())) {
            if (!TPackMgr::IsEnabled(os->GetName(), this)) {
               if (Load(os->GetName(), optls) < 0) {
                  // Package loading failure
                  Log(TString::Format("error: TPackMgr::Load: failure loading dep %s ...", os->GetName()));
                  return -1;
               }
            }
         }
      }
   }

   // Make sure it has been build
   Int_t chkveropt = kCheckROOT;
   if (optls) {
      TParameter<Int_t> *pcv = (TParameter<Int_t> *) optls->FindObject("PROOF_Package_CheckVersion");
      if (pcv) {
         chkveropt = pcv->GetVal();
         optls->Remove(pcv);
         delete pcv;
      }
   }
   if (Build(pack, chkveropt) < 0) {
      // Package not found
      Log(TString::Format("error: TPackMgr::Load: package %s oes not build! ", pack));
      return -1;
   }

   TString ocwd = gSystem->WorkingDirectory();
   gSystem->ChangeDirectory(pdir);

   // Shared lock from here
   TLockPathGuard lp(&fLock, kTRUE);

   // Check for SETUP.C and execute
   if (!gSystem->AccessPathName("PROOF-INF/SETUP.C")) {
      // We need to change the name of the function to avoid problems when we load more packages
      TString setup;
      setup.Form("SETUP_%d_%x", gSystem->GetPid(), TString(pack).Hash());
      // Remove special characters
      TMacro setupmc("PROOF-INF/SETUP.C");
      TObjString *setupline = setupmc.GetLineWith("SETUP(");
      if (setupline) {
         TString setupstring(setupline->GetString());
         setupstring.ReplaceAll("SETUP(", TString::Format("%s(", setup.Data()));
         setupline->SetString(setupstring);
      } else {
         // Macro does not contain SETUP()
         Log(TString::Format("warning: macro '%s/PROOF-INF/SETUP.C' does not contain a SETUP()"
                                         " function", pack));
      }

      // Load the macro
      if (!setupmc.Load()) {
         // Macro could not be loaded
         Log(TString::Format("error: macro '%s/PROOF-INF/SETUP.C' could not be loaded:"
                                " cannot continue", pack));
         rc = -1;
      } else {
         // Check the signature
         TFunction *fun = (TFunction *) gROOT->GetListOfGlobalFunctions()->FindObject(setup);
         if (!fun) {
            // Notify the upper level
            Log(TString::Format("error: function SETUP() not found in macro '%s/PROOF-INF/SETUP.C':"
                                   " cannot continue", pack));
            rc = -1;
         } else {
            TMethodCall callEnv;
            // Check the number of arguments
            if (fun->GetNargs() == 0) {
               // No arguments (basic signature)
               callEnv.Init(fun);
               if (optls && optls->GetSize() > 0) {
                  Log(TString::Format("warning: loaded SETUP() for '%s' does not take any argument:"
                                                " the specified argument will be ignored", pack));
               }
            } else if (fun->GetNargs() == 1) {
               TMethodArg *arg = (TMethodArg *) fun->GetListOfMethodArgs()->First();
               if (arg) {
                  callEnv.Init(fun);
                  // Check argument type
                  TString argsig(arg->GetTitle());
                  if (argsig.BeginsWith("TList")) {
                     callEnv.ResetParam();
                     callEnv.SetParam((Long_t) optls);
                  } else if (argsig.BeginsWith("const char")) {
                     callEnv.ResetParam();
                     TObjString *os = optls ? dynamic_cast<TObjString *>(optls->First()) : 0;
                     if (os) {
                        callEnv.SetParam((Long_t) os->GetName());
                     } else {
                        if (optls && optls->First()) {
                           Log(TString::Format("warning: found object argument of type %s:"
                                     " SETUP expects 'const char *': ignoring",
                                       optls->First()->ClassName()));
                        }
                        callEnv.SetParam((Long_t) 0);
                     }
                  } else {
                     // Notify the upper level
                     Log(TString::Format("error: unsupported SETUP signature: SETUP(%s)"
                               " cannot continue", arg->GetTitle()));
                     rc = -1;
                  }
               } else {
                  // Notify the upper level
                  Log("error: cannot get information about the SETUP() argument:"
                               " cannot continue");
                  rc = -1;
               }
            } else if (fun->GetNargs() > 1) {
               // Notify the upper level
               Log("error: function SETUP() can have at most a 'TList *' argument:"
                            " cannot continue");
               rc = -1;
            }
            // Execute
            Long_t setuprc = (rc == 0) ? 0 : -1;
            if (rc == 0) {
               callEnv.Execute(setuprc);
               if (setuprc < 0) rc = -1;
            }
         }
      }
   }

   gSystem->ChangeDirectory(ocwd);

   if (rc == 0) {
      // create link to package in working directory
      gSystem->Symlink(pdir, pack);

      // add package to list of include directories to be searched
      // by ACliC
      gSystem->AddIncludePath(TString::Format("-I%s", pack));

      // add package to list of include directories to be searched by CINT
      gROOT->ProcessLine(TString::Format(".I %s", pack));

      TPair *pck = (optls && optls->GetSize() > 0) ? new TPair(new TObjString(pack), optls->Clone())
                                                   : new TPair(new TObjString(pack), 0);
      if (!fEnabledPackages) {
         fEnabledPackages = new TList;
         fEnabledPackages->SetOwner();
      }
      fEnabledPackages->Add(pck);
   }

   return rc;
}


////////////////////////////////////////////////////////////////////////////////
/// Method to unload a package.
/// Return -1 on error, 0 otherwise

Int_t TPackMgr::Unload(const char *pack)
{
   Int_t rc = 0;

   if (fEnabledPackages && fEnabledPackages->GetSize() > 0) {
      TPair *ppack = 0;
      if (pack && strlen(pack) > 0) {
         if ((ppack = (TPair *) fEnabledPackages->FindObject(pack))) {

            // Remove entry from include path
            TString aclicincpath = gSystem->GetIncludePath();
            TString cintincpath = gInterpreter->GetIncludePath();
            // remove interpreter part of gSystem->GetIncludePath()
            aclicincpath.Remove(aclicincpath.Length() - cintincpath.Length() - 1);
            // remove package's include path
            aclicincpath.ReplaceAll(TString(" -I") + pack, "");
            gSystem->SetIncludePath(aclicincpath);

            //TODO reset interpreter include path

            // remove entry from enabled packages list
            delete fEnabledPackages->Remove(ppack);
         }

         // Cleanup the link, if there
         if (!gSystem->AccessPathName(pack))
            if (gSystem->Unlink(pack) != 0) rc = -1;

      } else {

         // Iterate over packages and remove each package
         TIter nxp(fEnabledPackages);
         while ((ppack = (TPair *) nxp())) {
            if (Unload(ppack->GetName()) != 0) rc = -1;
         }

      }
   }

   // We are done
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Method to check if this package manager has package 'pack'.
/// Return kTRUE or kFALSE

Bool_t TPackMgr::Has(const char *pack)
{
   // always follows BuildPackage so no need to check for PROOF-INF
   TString pdir = TString::Format("%s/%s", fDir.Data(), pack);

   // Shared lock from here
   TLockPathGuard lp(&fLock, kTRUE);

   if (gSystem->AccessPathName(pdir, kReadPermission) ||
       gSystem->AccessPathName(pdir + "/PROOF-INF", kReadPermission))
      return kFALSE;

   // Relevant directories exist and ar readable
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Method to check if 'path' is in the managed directory
/// Return kTRUE or kFALSE

Bool_t TPackMgr::IsInDir(const char *path)
{
   return strncmp(fDir.Data(), path, fDir.Length()) ? kFALSE : kTRUE ;
}

////////////////////////////////////////////////////////////////////////////////
/// Method to get the path of the dir for package 'pack'.
/// Return -1 in case of error (not found), 0 otherwise

Int_t TPackMgr::GetPackDir(const char *pack, TString &pdir)
{
   // Make sure the extension is not ".par"
   TString pn(pack);
   if (strstr(pack, ".par")) pn.Remove(pn.Last('.'));
   pdir.Form("%s/%s", fDir.Data(), pn.Data());
   if (gSystem->AccessPathName(pdir, kReadPermission)) return -1;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Method to get a semi-colon separated list with the names of the enabled
/// packages.

void TPackMgr::GetEnabledPackages(TString &packlist)
{
   packlist = "";
   if (!fEnabledPackages) return;

   TIter nxp(fEnabledPackages);
   TPair *pck= 0;
   while ((pck = (TPair *)nxp())) {
      if (packlist.Length() <= 0)
         packlist = pck->GetName();
      else
         packlist += TString::Format(";%s", pck->GetName());
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Method to get the path of the PAR file for package 'pack'.
/// Return -1 in case of error (not found), 0 otherwise

Int_t TPackMgr::GetParPath(const char *pack, TString &path)
{
   // Make sure the extension is ".par"
   const char *fm = (strstr(pack, ".par")) ? "%s/%s" : "%s/%s.par";
   path.Form(fm, fDir.Data(), pack);
   if (gSystem->AccessPathName(path, kReadPermission)) return -1;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Method to get the download dir; create if not existing
/// Return -1 in case of error (not found; not created), 0 otherwise

Int_t TPackMgr::GetDownloadDir(TString &dldir)
{
   dldir.Form("%s/downloaded", fDir.Data());
   if (gSystem->AccessPathName(dldir, kReadPermission)) {
      if (gSystem->mkdir(dldir, kTRUE) != 0) return -1;
      if (gSystem->AccessPathName(dldir, kReadPermission)) return -1;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Show available packages
///

void TPackMgr::Show(const char *title)
{
   if (fgGlobalPackMgrList && fgGlobalPackMgrList->GetSize() > 0) {
      // Scan the list of global packages dirs
      TIter nxpm(fgGlobalPackMgrList);
      TPackMgr *pm = 0;
      while ((pm = (TPackMgr *)nxpm())) {
         pm->Show(TString::Format("*** Global Package cache %s %s:%s ***\n",
                  pm->GetName(), gSystem->HostName(), pm->GetTitle()));
      }
   }

   if (title && strlen(title) > 0)
      printf("%s\n", title);
   else
      printf("*** Package cache %s:%s ***\n", gSystem->HostName(), fDir.Data());
   fflush(stdout);
   // Shared lock from here
   TLockPathGuard lp(&fLock, kTRUE);
   gSystem->Exec(TString::Format("%s %s", kLS, fDir.Data()));
   printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Clean dir for package 'pack'
/// Return -1 in case of error, 0 otherwise
///

Int_t TPackMgr::Clean(const char *pack)
{
   // Shared lock from here
   TLockPathGuard lp(&fLock);
   Int_t rc = 0;
   if (pack && strlen(pack)) {
      // remove package directory and par file
      rc = gSystem->Exec(TString::Format("%s %s/%s/*", kRM, fDir.Data(), pack));
   }
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove package 'pack'
/// If 'pack' is null or empty all packages are cleared
///

Int_t TPackMgr::Remove(const char *pack, Bool_t dolock)
{
   // Shared lock from here
   if (dolock) fLock.Lock();
   Int_t rc1 = 0, rc2 = 0, rc3 = 0;
   if (pack && strlen(pack)) {
      // remove package directory and par file
      TString path = TString::Format("%s/downloaded/%s.par", fDir.Data(), pack);
      gSystem->Exec(TString::Format("%s %s", kRM, path.Data()));
      if (!gSystem->AccessPathName(path, kFileExists)) rc1 = -1;
      path.ReplaceAll("/downloaded/", "/");
      gSystem->Exec(TString::Format("%s %s", kRM, path.Data()));
      if (!gSystem->AccessPathName(path, kFileExists)) rc2 = -1;
      path.Remove(path.Last('.'));
      gSystem->Exec(TString::Format("%s %s", kRM, path.Data()));
      if (!gSystem->AccessPathName(path, kFileExists)) rc3 = -1;
   } else {
      // Clear all packages
      rc1 = gSystem->Exec(TString::Format("%s %s/*", kRM, fDir.Data()));
   }
   if (dolock) fLock.Unlock();
   return (rc1 + rc2 + rc3);
}

////////////////////////////////////////////////////////////////////////////////
/// Get list of available packages
/// Returns a pointer to a TList object, transferring ownership to the caller

TList *TPackMgr::GetList() const
{
   TList *plist = new TList;
   void *dir = gSystem->OpenDirectory(fDir);
   if (dir) {
      TString pac(gSystem->GetDirEntry(dir));
      while (pac.Length() > 0) {
         if (pac.EndsWith(".par")) {
            pac.ReplaceAll(".par","");
            plist->Add(new TObjString(pac.Data()));
         }
         pac = gSystem->GetDirEntry(dir);
      }
   }
   gSystem->FreeDirectory(dir);

   return plist;
}

////////////////////////////////////////////////////////////////////////////////
/// Get list of enabled packages
/// Returns a pointer to a TList object, transferring ownership to the caller

TList *TPackMgr::GetListOfEnabled() const
{
   TList *epl = nullptr;
   if (fEnabledPackages && fEnabledPackages->GetSize() > 0) {
      epl = new TList;
      TIter nxp(fEnabledPackages);
      TObject *o = 0;
      while ((o = nxp())) {
         epl->Add(new TObjString(o->GetName()));
      }
   }
   return epl;
}

////////////////////////////////////////////////////////////////////////////////
/// Show enabled packages
///

void TPackMgr::ShowEnabled(const char *title)
{
   if (fgGlobalPackMgrList && fgGlobalPackMgrList->GetSize() > 0) {
      // Scan the list of global packages dirs
      TIter nxpm(fgGlobalPackMgrList);
      TPackMgr *pm = 0;
      while ((pm = (TPackMgr *)nxpm())) {
         pm->ShowEnabled(TString::Format("*** Global Package cache %s %s:%s ***\n",
                  pm->GetName(), gSystem->HostName(), pm->GetTitle()));
      }
   }

   if (!fEnabledPackages || fEnabledPackages->GetSize() <= 0) return;

   if (title && strlen(title) > 0)
      printf("%s\n", title);
   else
      printf("*** Package enabled on %s ***\n", gSystem->HostName());
   fflush(stdout);

   TIter next(fEnabledPackages);
   while (TPair *pck = (TPair *) next()) {
      printf("%s\n", pck->GetName());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get MD5 checksum of the PAR file corresponding to given package
/// Returns a pointer to a TMD5 object, transferring ownership to the caller

TMD5 *TPackMgr::GetMD5(const char *pack)
{
   // Shared lock from here
   TLockPathGuard lp(&fLock, kTRUE);
   // PAR file path
   const char *fm = (strstr(pack, ".par")) ? "%s/%s" : "%s/%s.par";
   TString parfile = TString::Format(fm, fDir.Data(), pack);

   return TMD5::FileChecksum(parfile);
}


////////////////////////////////////////////////////////////////////////////////
/// Read MD5 checksum of the PAR file from the PROOF-INF/md5.txt file.
/// Returns a pointer to a TMD5 object, transferring ownership to the caller

TMD5 *TPackMgr::ReadMD5(const char *pack)
{
   TString pn(pack);
   if (pn.EndsWith(".par")) pn.Remove(pn.Last('.'));

   TString md5f = TString::Format("%s/%s/PROOF-INF/md5.txt", fDir.Data(), pn.Data());
   TLockPathGuard lp(&fLock, kTRUE);
   return TMD5::ReadChecksum(md5f);
}


////////////////////////////////////////////////////////////////////////////////
/// Read MD5 checksum of the PAR file from the PROOF-INF/md5.txt file.
/// Returns a pointer to a TMD5 object, transferring ownership to the caller

Int_t TPackMgr::Unpack(const char *pack, TMD5 *sum)
{
   Int_t rc = 0;
   TString fn(pack), pn(pack);
   if (!fn.EndsWith(".par")) fn += ".par";
   if (pn.EndsWith(".par")) pn.Remove(pn.Last('.'));

   // Find gunzip...
   char *gunzip = gSystem->Which(gSystem->Getenv("PATH"), kGUNZIP, kExecutePermission);
   if (gunzip) {
      // untar package
      TString cmd;
      cmd.Form(kUNTAR, gunzip, fDir.Data(), fn.Data(), fDir.Data());
      rc = gSystem->Exec(cmd);
      if (rc != 0)
         Error("Unpack", "failure executing: %s (rc: %d)", cmd.Data(), rc);
      delete [] gunzip;
   } else {
      Error("Unpack", "%s not found", kGUNZIP);
      rc = -2;
   }
   // check that fDir/pack now exists
   if (gSystem->AccessPathName(TString::Format("%s/%s", fDir.Data(), pn.Data()), kWritePermission)) {
      // par file did not unpack itself in the expected directory, failure
      rc = -1;
      Error("Unpack", "package %s did not unpack into %s", fn.Data(), pn.Data());
   } else {
      // store md5 in package/PROOF-INF/md5.txt
      if (sum) {
         TString md5f = TString::Format("%s/%s/PROOF-INF/md5.txt", fDir.Data(), pn.Data());
         TMD5::WriteChecksum(md5f, sum);
      }
   }

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Install package from par (unpack the file in the directory); par can be an
/// URL for remote retrieval. If rmold is kTRUE an existing version of the package
/// is removed if existing.
/// Returns 0 on success, <0 otherwise

Int_t TPackMgr::Install(const char *parpath, Bool_t rmold)
{
   Int_t rc = 0;

   Info("Install", "installing %s ...", parpath);
   TString par = parpath;
   gSystem->ExpandPathName(par);

   // Does par exists?
   if (gSystem->AccessPathName(par.Data(), kReadPermission)) {
      Error("Install", "%s is invalid", par.Data());
      return -1;
   }
   TString parname = gSystem->BaseName(par.Data());
   TString pack = parname(0, parname.Last('.'));
   TString dest = TString::Format("%s/%s", fDir.Data(), parname.Data());
   TString psrc = par, ssrc;
   TMD5 *sums = 0, *md5 = 0, *md5d = 0;

   // Check if we need to download: get the remote checksum
   // Retrieve the checksum of the file, if available
   // Dowload checksum file, if available
   TString dldir;
   if (GetDownloadDir(dldir) != 0) {
      Error("Install", "could not create/get download directory");
      return -1;
   }

   TLockPathGuard lp(&fLock, kFALSE);

   TString parsum(par);
   parsum.ReplaceAll(".par", ".md5sum");
   if (!gSystem->AccessPathName(parsum, kReadPermission)) {
      ssrc.Form("%s/%s", dldir.Data(), gSystem->BaseName(parsum));
      if (!TFile::Cp(parsum, ssrc)) {
         Warning("Install", "could not retrieve %s", parsum.Data());
      } else {
         md5 = TMD5::ReadChecksum(ssrc);
      }
   }

   // Do we have already the file?
   Bool_t parexists = (!gSystem->AccessPathName(dest)) ? kTRUE : kFALSE;

   Bool_t install = kTRUE;
   // If yes and we are asked to clean the old one, do it
   if (parexists) {
      install = kFALSE;
      if (rmold) {
         // Asked to remove: do it
         if (Remove(pack, kFALSE) < 0) {
            Error("Install", "could not remove existing version of '%s'", pack.Data());
            if (md5) delete md5;
            return -1;
         }
         install = kTRUE;
      } else {
         if (!md5) {
            TFile::EFileType ft = TFile::GetType(par.Data());
            if (ft == TFile::kWeb || ft == TFile::kNet) {
               psrc.Form("%s/%s", dldir.Data(), parname.Data());
               if (!TFile::Cp(par.Data(), psrc)) {
                  Error("Install", "could not retrieve %s", par.Data());
                  return -1;
               }
            }
            // psrc is either the original par or the downloaded path
            md5 = TMD5::FileChecksum(psrc);
         }
         // Now we need to compare with the local one
         sums = TMD5::FileChecksum(dest);
         if (sums && md5 && (*sums != *md5)) install = kTRUE;
      }
   }
   if (sums) delete sums;

   // Install if required
   if (install) {
      if (!TFile::Cp(psrc, dest)) {
         Error("Install", "could not copy %s to %s", psrc.Data(), dest.Data());
         if (md5) delete md5;
         return -1;
      }
   }
   md5d = TMD5::FileChecksum(dest);

   if (md5 && *md5 != *md5d)
      Warning("Install", "checksums do not match:\n\tdownloaded:\t%s\n\texpected:\t%s",
                         md5d->AsString(), md5->AsString());
   if (Unpack(pack, md5d) != 0) {
      Error("Install", "could not unpack %s", dest.Data());
      rc = -1;
   }
   if (md5) delete md5;
   if (md5d) delete md5d;
   return rc;
}

//---------------------------------------------------------------------------------------------------
//    Static methods
//---------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
/// Parse one or more paths as possible sources of packages
/// Returns number of paths added; or -1 in case of problems

Int_t TPackMgr::RegisterGlobalPath(const char *paths)
{
   Int_t ng = 0;
   // List of directories where to look for global packages
   TString globpack(paths);
   if (globpack.Length() > 0) {
      Int_t from = 0;
      TString ldir;
      while (globpack.Tokenize(ldir, from, ":")) {
         if (gSystem->AccessPathName(ldir, kReadPermission)) {
            ::Warning("TPackMgr::RegisterGlobalPath",
                      "directory for global packages %s does not"
                      " exist or is not readable", ldir.Data());
         } else {
            // Add to the list, key will be "G<ng>", i.e. "G0", "G1", ...
            TString key;
            key.Form("G%d", ng++);
            if (!fgGlobalPackMgrList) {
               fgGlobalPackMgrList = new THashList();
               fgGlobalPackMgrList->SetOwner();
            }
            TPackMgr *pmgr = new TPackMgr(ldir);
            pmgr->SetName(key);
            fgGlobalPackMgrList->Add(pmgr);
            ::Info("TPackMgr::RegisterGlobalPath",
                   "manager for global packages directory %s added to the list",
                   ldir.Data());
         }
      }
   }
   // Number of registered packages
   return ng;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the package manager having 'pack'; priority is given to packmgr, if
/// defined.
/// Returns packmgr or nullptr

TPackMgr *TPackMgr::GetPackMgr(const char *pack, TPackMgr *packmgr)
{
   if (packmgr && packmgr->Has(pack)) return packmgr;

   if (fgGlobalPackMgrList && fgGlobalPackMgrList->GetSize() > 0) {
      // Scan the list of global packages managers
      TIter nxpm(fgGlobalPackMgrList);
      TPackMgr *pm = 0;
      while ((pm = (TPackMgr *)nxpm())) {
         if (pm->Has(pack)) return pm;
      }
   }
   return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the full path to PAR, looking also in the global dirs.
/// Returns -1 if not found, 0 if available in global dirs, 1 if it can be
/// uploaded from the local package dir.
/// For the cases >= 0, par is filled with the path of the PAR file

Int_t TPackMgr::FindParPath(TPackMgr *packmgr, const char *pack, TString &par)
{
   // Try the package dir
   if (packmgr && packmgr->GetParPath(pack, par) == 0) return 1;

   // Try global package dirs
   if (fgGlobalPackMgrList && fgGlobalPackMgrList->GetSize() > 0) {
      // Scan the list of global packages dirs
      TIter nxpm(fgGlobalPackMgrList);
      TPackMgr *pm = 0;
      while ((pm = (TPackMgr *)nxpm())) {
         if (pm->GetParPath(pack, par) == 0) {
            // Package found, stop searching
            break;
         }
         par = "";
      }
      if (par.Length() > 0) return 0;
   }
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the package is enabled; priority is given to packmgr, if
/// defined.
/// Returns kTRUE if enabled

Bool_t TPackMgr::IsEnabled(const char *pack, TPackMgr *packmgr)
{
   if (packmgr && packmgr->IsPackageEnabled(pack)) return kTRUE;

   if (fgGlobalPackMgrList && fgGlobalPackMgrList->GetSize() > 0) {
      // Scan the list of global packages managers
      TIter nxpm(fgGlobalPackMgrList);
      TPackMgr *pm = 0;
      while ((pm = (TPackMgr *)nxpm())) {
         if (pm->IsPackageEnabled(pack)) return kTRUE;
      }
   }
   // Not Enabled
   return kFALSE;
}
