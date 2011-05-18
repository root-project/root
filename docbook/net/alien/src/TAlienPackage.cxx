// @(#)root/alien:$Id$
// Author: Lucia Jancurova/Andreas-Joachim Peters 1/10/2007

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TSystem.h"
#include "TGridResult.h"
#include "TFile.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TError.h"
#include "TAlienPackage.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienPackage                                                        //
//                                                                      //
// Class providing package management functionality like the AliEn      //
// Package Management System.                                           //
// Allows to setup software packages on a local desktop like in the     //
// GRID environment and to execute the contained programs.              //
// Registered Dependencies are automatically resolved and missing       //
// packages are automatically installed.                                //
// Currently there is no support for 'source' packages.                 //
// The desired platform has to be specified in the constructor.         //
// The default constructor takes packages from the global package       //
// section in AliEn. If you want to install a user package, you have to //
// set the AliEn package directory to your local package directory using//
// 'package->SetAliEnMainPackageDir("/alice/cern.ch/user/..../packages")//
//                                                                      //
// ---------------------------------------------------------------------/////////////////////////
// Examples of use:                                                                            //
// root [0] TAlienPackage* package = new TAlienPackage("AliRoot","v4-07-Rev-01","Linux-i686"); //
// root [1] package->Exec("aliroot -b -q ")                                                    //
//                                                                                             //
// root [0] TAlienPackage* package = new TAlienPackage("ROOT","v5-16-00","Linux-i686");        //
// root [1] package->Exec("root -b -q ")                                                       //
/////////////////////////////////////////////////////////////////////////////////////////////////



ClassImp (TAlienPackage)

//______________________________________________________________________________
TAlienPackage::TAlienPackage() : fInstallList (0), fPackages (0), fDebugLevel (0)
{
   // Default constructor of a AliEn package constructing a ROOT:v5-16-00 for Linux-i686.

   fName = "ROOT";
   fVersion = "v5-16-00";
   fPlatform = "Linux-i686";
   fAliEnMainPackageDir = "/alice/packages";
   fInstallationDirectory = "/var/tmp/alien/packages";
   fPostInstallCommand = "post_install";
   fEnableCommand = "";

   if (gDebug > 0)
      Info ("TAlienPackage",
            "\tPackage=%s Version=%s Platform=%s Installdir=%s AlienInstalldir=%s PostInstall=%s",
            fName.Data (), fVersion.Data (), fPlatform.Data (),
            fInstallationDirectory.Data (), fAliEnMainPackageDir.Data (),
            fPostInstallCommand.Data ());
   if (!gGrid)
      gGrid = TGrid::Connect ("alien://");

   if (!fInstallList)
      fInstallList = new TList ();
   if (!fPackages)
      fPackages = new TList ();

   fEnabled = kFALSE;
   fPackages->SetOwner (kFALSE);
}

//______________________________________________________________________________
TAlienPackage::TAlienPackage(const char *name, const char *version,
                             const char *platform,
                             const char *installationdirectory) :
   fInstallList (0), fPackages (0), fDebugLevel (0)
{
   // Constructor of a AliEn package.

   fName = name;

   fVersion = version;

   fAliEnMainPackageDir = "/alice/packages";
   fInstallationDirectory = installationdirectory;
   fPostInstallCommand = "post_install";
   fEnableCommand = "";
   fPlatform = platform;
   fEnabled = kFALSE;

   if (gDebug > 0)
      Info ("TAlienPackage",
            "\tPackage=%s Version=%s Platform=%s Installdir=%s AlienInstalldir=%s PostInstall=%s",
            name, version, platform, installationdirectory,
            fAliEnMainPackageDir.Data (), fPostInstallCommand.Data ());

   if (!gGrid)
      gGrid = TGrid::Connect ("alien://");

   if (!fInstallList)
      fInstallList = new TList ();

   if (!fPackages)
      fPackages = new TList ();
   fPackages->SetOwner (kFALSE);
}

//______________________________________________________________________________
TAlienPackage::~TAlienPackage()
{
   // Destructor.

   if (GetDebugLevel () > 2)
      Info ("~TAlienPackage", "\tDestr: Package=%s Version=%s Platform=%s",
            fName.Data (), fVersion.Data (), fPlatform.Data ());
   SafeDelete (fInstallList);
}

//______________________________________________________________________________
Bool_t TAlienPackage::Enable()
{
   // Install/enable an AliEn package on the local computer.

   fInstallList->Clear ();
   if (GetDebugLevel () > 1)
      Info ("Install", "\t\tInstalling Package=%s Version=%s Platform=%s",
            fName.Data (), fVersion.Data (), fPlatform.Data ());

   if (CheckDirectories (fName, fVersion) == kFALSE)
      return kFALSE;

   if (CheckDependencies () == kFALSE)
      return kFALSE;

   if (InstallAllPackages () == kFALSE)
      return kFALSE;

   gSystem->Exec(Form("mkdir -p %s/%s/%s/%s ; touch  %s/%s/%s/%s/.safeguard",
                 fInstallationDirectory.Data (), fName.Data (), fVersion.Data (),
                 fVersion.Data (), fInstallationDirectory.Data (), fName.Data (),
                 fVersion.Data (), fVersion.Data ()));

   fEnabled = kTRUE;
   return kTRUE;
}

//______________________________________________________________________________
const char *TAlienPackage::GetEnable ()
{
   // Return shell command to enable package.

   fEnableCommand =
      Form ("%s/%s/%s/.alienEnvironment %s/%s/%s ",
            fInstallationDirectory.Data (), fName.Data (), fVersion.Data (),
            fInstallationDirectory.Data (), fName.Data (), fVersion.Data ());
   return fEnableCommand.Data ();
}

//______________________________________________________________________________
Bool_t TAlienPackage::UnInstall ()
{
   // Uninstall a package e.g. remove it from the local disk.

   gSystem->Exec(Form
            ("test -e %s/%s/%s/%s/.safeguard && rm -rf %s/%s/%s",
             fInstallationDirectory.Data (), fName.Data (), fVersion.Data (),
             fVersion.Data (), fInstallationDirectory.Data (), fName.Data (),
             fVersion.Data ()));
   fEnabled = kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TAlienPackage::IsDirectory (const char *dir1, const char *str)
{
   // Check that <str> is listed in GRID directory <dir1>.

   TGridResult *result = gGrid->Ls (dir1);
   Int_t i = 0;
   while (result->GetFileName (i)) {
      if (TString (result->GetFileName (i)) == str) {
         return kTRUE;
      }
      i++;
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TAlienPackage::CheckDirectories (TString name, TString version)
{
   // Check the name and version directory of package/version given.

   TString s(GetAliEnMainPackageDir());

   if ((IsDirectory (s.Data (), name.Data ())) == kTRUE) {
      if (GetDebugLevel () > 1)
         Info ("CheckDirectories", "\t%s/%s exists.", s.Data (), name.Data ());

      s += "/" + name;
      if ((IsDirectory (s, version.Data ())) == kTRUE) {
         if (GetDebugLevel () > 1)
            Info ("CheckDirectories", "\t%s/%s exist.", s.Data (), version.Data ());

         s += "/" + version;
         if ((IsDirectory (s, GetPlatform ().Data ())) == kTRUE) {
            if (GetDebugLevel () > 1)
               Info ("CheckDirectories", "\t%s/%s exist.", s.Data (), GetPlatform ().Data ());
            return kTRUE;
         } else {
            Error ("CheckDirectories", "\t%s/%s does not exist.", s.Data (), GetPlatform ().Data ());
         }
      } else {
         Error ("CheckDirectories", "\t%s/%s does not exist.", s.Data (), version.Data ());
      }
   }  else {
      Info ("CheckDirectories", "\t%s/%s exists.", s.Data (), name.Data ());
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TAlienPackage::ReInstall ()
{
   // Reinstalls a package e.g. uninstall + install.

   if (UnInstall () == kFALSE)
      return kFALSE;

   if (Enable () == kFALSE)
      return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TAlienPackage::PostInstall (TString name, TString version)
{
   // Execute post_install procedure for a package.

   TGridResult *result =
      gGrid->Command (Form
               ("showTagValue -z %s/%s/%s PackageDef",
                fAliEnMainPackageDir.Data (), name.Data (), version.Data ()));
   TString post_install (result->GetKey (0, "post_install"));

   if (post_install.IsNull () == kTRUE) {
      if (GetDebugLevel () > 0)
         Info ("PostInstall",
               "\tNo post install procedure defined in AliEn.");
      return kTRUE;
   }

   if (GetDebugLevel () > 0)
      Info ("PostInstall",
            "\tDownloading PostInstall for Package=%s Version=%s",
            name.Data (), version.Data ());

   if (!TFile::Cp(Form("alien://%s", post_install.Data ()),
                  Form("%s/%s/%s/%s", fInstallationDirectory.Data (), name.Data (),
                  version.Data (), fPostInstallCommand.Data ()))) {
      Error ("PostInstall", "\tCannot download the PostInstall script %s!", post_install.Data ());
      return kFALSE;
   }

   gSystem->ChangeDirectory (Form
                       ("%s/%s/%s", fInstallationDirectory.Data (),
                        name.Data (), version.Data ()));
   gSystem->Exec (Form ("chmod +x %s", fPostInstallCommand.Data ()));
   gSystem->Exec (Form ("./%s %s/%s/%s", fPostInstallCommand.Data (),
             fInstallationDirectory.Data (), name.Data (), version.Data ()));

   if (GetDebugLevel () > 1)
      Info ("PostInstall",
            "\tExecuted PostInstall for Package=%s Version=%s ", name.Data (),
            version.Data ());
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TAlienPackage::Exec (const char *cmdline)
{
   // Execute package command.

   TString fullline = "";

   if (!fEnabled) {
      if (!Enable ())
         return kFALSE;
   }

   for (Int_t j = 0; j < fPackages->GetEntries (); j++) {
      TAlienPackage *package = (TAlienPackage *) fPackages->At (j);
      fullline += package->GetEnable ();
      fullline += " ";
   }

   fullline += cmdline;

   Info("Exec", "\t\tExecuting Package=%s Version=%s \"%s\"", fName.Data (),
        fVersion.Data (), fullline.Data ());

   gSystem->Exec(fullline.Data());
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TAlienPackage::CheckDependencies ()
{
   // Check the dependency packages of this package.

   TString path (Form("%s/%s/%s", fAliEnMainPackageDir.Data (), fName.Data (),
             fVersion.Data ()));

   TGridResult *result =
      gGrid->Command (Form ("showTagValue -z %s PackageDef", path.Data ()));

   TString strDep (result->GetKey (0, "dependencies"));

   if (strDep.IsNull () == kTRUE) {
      if (GetDebugLevel () > 0)
         Info ("CheckDepencencies", "\tFound no dependencies ... ");
      TObjString *strObj =
         new TObjString (Form ("%s::%s", fName.Data (), fVersion.Data ()));
      fInstallList->Add (strObj);
      return kTRUE;
   }

   TObjArray *strDeps = strDep.Tokenize (",");

   if (GetDebugLevel () > 0)
      Info ("CheckDepencencies", "\tFound %d dependencies ... ",
            strDeps->GetEntries ());

   for (Int_t i = 0; i < strDeps->GetEntries (); i++) {
      TObjString *strObj = (TObjString *) strDeps->At (i);
      TObjArray *strDepsPackgAndVer = strObj->GetString ().Tokenize ("@");
      TObjString *strObj2 = (TObjString *) strDepsPackgAndVer->At (1);

      if (GetDebugLevel () > 2)
         Info ("CheckDependencies", "\t[%d] Dep. Package=%s", i,
               strObj2->GetString ().Data ());
      fInstallList->Add (strObj2);
   }

   TObjString *strObj = new TObjString (Form ("%s::%s", fName.Data (), fVersion.Data ()));
   fInstallList->Add (strObj);

   for (Int_t j = 0; j < fInstallList->GetEntries (); j++) {
      strObj = (TObjString *) fInstallList->At(j);
      TString strObjPackage, strObjVersion;
      Int_t from = 0;
      if (strObj->GetString().Tokenize(strObjPackage, from, "::")) {
         if (!strObj->GetString().Tokenize(strObjVersion, from, "::")) {
            Warning("CheckDepencencies", "version string not found for j=%d (%s)", j, strObj->GetName());
            continue;
         }
      } else {
         Warning("CheckDepencencies", "package string not found for j=%d (%s)", j, strObj->GetName());
         continue;
      }

      if (GetDebugLevel () > 2)
         Info ("CheckDepencencies", "\t[%d] Name=%s Version=%s", j,
               strObjPackage.Data(), strObjVersion.Data());

      if (CheckDirectories(strObjPackage, strObjVersion) == kFALSE)
         return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TAlienPackage::InstallSinglePackage(TString name, TString version, Bool_t isDep)
{
   // Install a single package.

   Info ("InstallSinglePackage", "\t%s %s", name.Data (), version.Data ());
   // install a package without dependencies
   TString s1 (Form ("%s/%s/%s/%s", fAliEnMainPackageDir.Data (), name.Data (),
           version.Data (), fPlatform.Data ()));
   TString s2 (Form ("%s(%s)", name.Data (), version.Data ()));
   TString s3 (Form ("%s/%s/%s/%s", fInstallationDirectory.Data (), name.Data (),
           version.Data (), version.Data ()));
   TString s4 (Form ("%s/%s/%s/%s/%s", fInstallationDirectory.Data (), name.Data (),
           version.Data (), version.Data (), fPlatform.Data ()));
   TString s5 (Form ("%s/%s/%s/%s", fInstallationDirectory.Data (), name.Data (),
           version.Data (), fPlatform.Data ()));
   TString s6 (Form ("%s/%s", fAliEnMainPackageDir.Data (), name.Data ()));
   TString s7 (Form ("%s/%s/%s", fInstallationDirectory.Data (), name.Data (),
           version.Data ()));
   TString s8 (Form ("%s/%s/%s/%s/.safeguard", fInstallationDirectory.Data (),
           name.Data (), version.Data (), version.Data ()));

   if (gSystem->AccessPathName (s8.Data ()) == 0) {
      if (isDep == kFALSE) {
         if (GetDebugLevel () > 0) {
            Warning ("InstallSinglePackage",
                     "\tPackage=%s exists in /%s directory.",
                     s2.Data (), s3.Data ());
            Warning ("InstallSinglePackage",
                     "\tYou might use function UnInstall() before Enable(), or do ReInstall() !!!!");
         }
         return kTRUE;
      } else {
         return kTRUE;
      }
   }

   if (GetDebugLevel () > 1)
      Info ("InstallSinglePackage", "\tCopying from alien://%s to %s ",
            s1.Data (), s5.Data ());

   gSystem->Exec (Form ("mkdir -p %s", s3.Data ()));

   if (gSystem->AccessPathName (s3.Data ())) {
      Error ("InstallSinglePackage", "\tCouldn't create directory %s !",
             s3.Data ());
      return kFALSE;
   }

   if (!TFile::Cp (Form ("alien://%s", s1.Data ()), Form ("%s", s5.Data ()))) {
      Error ("InstallSinglePackage", "\tCouldn't copy alien://%s -> %s",
             s1.Data (), s5.Data ());
      return kFALSE;
   }

   if (GetDebugLevel () > 2)
      Info ("InstallSinglePackage", "\tEntering directory %s ", s7.Data ());

   if (!gSystem->ChangeDirectory (Form ("%s", s7.Data ()))) {
      Error ("InstallSinglePackage", "\tCannot change into directory %s",
             s7.Data ());
      return kFALSE;
   }

   if (GetDebugLevel () > 2)
      Info ("InstallSinglePackage", "\tUnpacking the package %s ...",
            s2.Data ());

   gSystem->Exec (Form ("tar -xzf %s", fPlatform.Data ()));

   if (GetDebugLevel () > 2)
      Info ("InstallSinglePackage", "\tUnpacking the package %s DONE ...",
            s2.Data ());

   gSystem->Exec (Form ("rm -f %s", fPlatform.Data ()));

   if (GetDebugLevel () > 2)
      Info ("InstallSinglePackage",
            "\tCopying PostInstall alien://%s/%s -> %s", s6.Data (),
            fPostInstallCommand.Data (), s7.Data ());

   if (!PostInstall (name, version)) {
      Error ("InstallSinglePackage",
             "\tPostInstall procedure failed for package %s failed!",
             s2.Data ());
      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TAlienPackage::InstallAllPackages ()
{
   // Installs a package and all its direct dependencies.

   Bool_t isDep = kFALSE;

   Info ("InstallAllPackages", "\tPackage=%s Version=%s", fName.Data (),
         fVersion.Data ());

   for (Int_t j = 0; j < fPackages->GetEntries (); j++) {
      TAlienPackage *package = (TAlienPackage *) fPackages->At (j);
      if (package && (package != this))
         delete package;
   }

   fPackages->Clear ();

   for (Int_t j = 0; j < fInstallList->GetEntries (); j++) {
      TObjString *strObj = (TObjString *) fInstallList->At (j);

      TObjArray *strDepsPackgOrVer = strObj->GetString ().Tokenize ("::");
      TObjString *strObjPackage = (TObjString *) strDepsPackgOrVer->At (0);
      TObjString *strObjVersion = (TObjString *) strDepsPackgOrVer->At (1);
      if (GetDebugLevel () > 1)
         Info ("InstallAllPackages", "\tPackage=%s Version=%s",
               strObjPackage->GetString ().Data (),
               strObjVersion->GetString ().Data ());

      if (j < (fInstallList->GetEntries () - 1))
         isDep = kTRUE;
      else
         isDep = kFALSE;

      if (j == (fInstallList->GetEntries () - 1)) {
         if (InstallSinglePackage(strObjPackage->GetString (), strObjVersion->GetString (), isDep) == kFALSE)
            return kFALSE;
         fPackages->Add ((TObject *) this);
      } else {
         TAlienPackage *newpackage = new TAlienPackage(strObjPackage->GetName(),
                                                       strObjVersion->GetName(),
                                                       fPlatform.Data());
         if (newpackage) {
            if (!newpackage->Enable ())
               return kFALSE;
         } else {
            return kFALSE;
         }

         fPackages->Add ((TObject *) newpackage);
      }
   }

   return kTRUE;
}
