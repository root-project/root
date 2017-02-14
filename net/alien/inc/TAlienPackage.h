// @(#)root/alien:$Id$
// Author: Lucia Jancurova/Andreas-Joachim Peters 1/10/2007

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienPackage
#define ROOT_TAlienPackage

#include "Rtypes.h"
#include "TList.h"
#include "TGrid.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienPackage                                                        //
//                                                                      //
// Class providing package management functionality like the AliEn      //
// Package Management System.                                           //
// Allows to setup software packages on a local desktop like in the     //
// GRID environment.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TAlienPackage {

private:
   TString fName;                   // package principal name
   TString fVersion;                // package version
   TString fPlatform;               // package platform
   TString fInstallationDirectory;  // installation directory
   TString fAliEnMainPackageDir;    // path to alien packages in the AliEn FC
   TString fPostInstallCommand;     // command to execute for postinstall
   TString fEnableCommand;          // command to enable the package
   TList  *fInstallList;            // package list with names of dependency packages
   TList  *fPackages;               // package list TPackage with this and dependency packages
   Int_t  fDebugLevel;              // internal debug level
   Bool_t fEnabled;                 // true if package is enabled for execution

   Bool_t PostInstall (TString name, TString version);                                 // runs the postinstall procedure for this package
   Bool_t InstallSinglePackage (TString name, TString version, Bool_t isDep = kFALSE); // installs the defined package without dependencies
   Bool_t InstallAllPackages (); // installs the defined package + all dependency packages

 public:
   TAlienPackage ();

   TAlienPackage(const char *name,
                 const char *version,
                 const char *platform,
                 const char *installationdirectory = "/var/tmp/alien/packages");

   virtual ~ TAlienPackage ();

   Bool_t Enable();            // install/enable the defined package
   const char *GetEnable();    // return shell command to enable package
   Bool_t Exec(const char *cmdline);  // execute <cmd> with this package
   Bool_t UnInstall();         // uninstall the defined package
   Bool_t ReInstall();         // reinstall the defined package
   Bool_t CheckDependencies(); // get all the dependencies of a package

   Bool_t IsDirectory(const char *dir1, const char *str);  // check for <str> in GRID directory <dir1>
   Bool_t CheckDirectories(TString name, TString version); // check that the defined package is existing as an alien package directory

   void SetName(const TString & theValue) { fName = theValue; }
   // set the name of the package
   TString GetName() const { return fName; }
   // get the name of the package
   void SetVersion(const TString & theValue) { fVersion = theValue; }
   // set the version of the package
   TString GetVersion() const { return fVersion; }
   // get the version of the package
   void SetPlatform(const TString & theValue) { fPlatform = theValue; }
   // set the platform for the package
   TString GetPlatform() const { return fPlatform; }
   // get the platform for the package
   void SetInstallationDirectory(const TString & theValue) { fInstallationDirectory = theValue; }
   // set the installation directory
   TString GetInstallationDirectory() const { return fInstallationDirectory; }
   // get the installation directory
   void SetAliEnMainPackageDir(const TString & theValue) { fAliEnMainPackageDir = theValue; }
   // set the alien path to look for the named package
   TString GetAliEnMainPackageDir() const { return fAliEnMainPackageDir; }
   // get the alien path to look for the named package
   void SetInstallList(TList * theValue) { fInstallList = theValue; }
   // set the install(dependency) package list
   TList *GetInstallList() const { return fInstallList; }
   // get the install(dependency) package list;
   void SetDebugLevel(Int_t & theValue) { fDebugLevel = theValue; }
   // set the internal debug level
   Int_t GetDebugLevel() { return fDebugLevel; }
   // get the internal debug level

   ClassDef (TAlienPackage, 0);   // Alien package interface
};

#endif
