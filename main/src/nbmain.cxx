// @(#)root/main:$Id$
// Author: Enric Tejedor   07/10/15

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// NBMain                                                               //
//                                                                      //
// Main program used to spawn a ROOT notebook                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"

#include "TCollection.h"
#include "TList.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TSystemDirectory.h"
#include "TSystemFile.h"

#include <fstream>
#include <string>

#define IPYTHON_CMD     "ipython"
#define NB_OPT          "notebook"
#define IPYTHON_DIR_VAR "IPYTHONDIR"
#define NB_CONF_DIR     "notebook"


using namespace std;

#ifdef WIN32
static string pathsep("\\");
#else
static string pathsep("/");
#endif
static string rootnbdir(".rootnb" + pathsep);
static string ipyconfigpath("profile_default" + pathsep + "ipython_notebook_config.py");
static string commitfile(".rootcomit");

////////////////////////////////////////////////////////////////////////////////
/// Checks whether ROOT notebook files are installed and they are
/// the current version.

static int CheckNbInstallation(string dir)
{
   string commit(gROOT->GetGitCommit());
   string inputfname(dir + pathsep + rootnbdir + commitfile);
   ifstream in(inputfname);
   if (in.is_open()) {
      string line;
      in >> line;
      in.close();
      if (line.compare(commit) == 0) return  0; // already installed
      else                           return -1; // install, it's outdated
   }
   else if (gSystem->AccessPathName(inputfname.c_str())) {
      // There is no installation
      return -1;
   }
   else {
      fprintf(stderr,
              "Error checking notebook installation -- cannot open %s\n",
              inputfname.c_str());
      return -2;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Installs ROOT notebook files in the user's home directory.

static bool InstallNbFiles(string source, string dest)
{
   // Create installation directory
   if (gSystem->AccessPathName(dest.c_str())) {
      if (gSystem->mkdir(dest.c_str())) {
         fprintf(stderr,
                 "Error installing notebook configuration files -- cannot create directory %s\n",
                 dest.c_str());
         return false;
      }
   }

   // Copy files in source to dest
   TSystemDirectory dir(source.c_str(), source.c_str());
   TList *files = dir.GetListOfFiles();
   if (files) {
      TSystemFile *file;
      TListIter it(files);
      while ((file = (TSystemFile*)it())) {
         TString s = file->GetName();
         string fname(s.Data());
         string sourcefile = source + pathsep + fname;
         string destfile   = dest   + pathsep + fname;
         if (!file->IsDirectory()) {
            if (gSystem->CopyFile(sourcefile.c_str(), destfile.c_str(), true)) {
               fprintf(stderr,
                       "Error installing notebook configuration files -- cannot copy file %s to %s\n",
                       sourcefile.c_str(), destfile.c_str());
               return false;
            }
         }
         else if (fname.compare(".") && fname.compare("..")) {
            if (!InstallNbFiles(sourcefile, destfile))
               return false;
         }
      }
   }

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates the IPython notebook configuration file that sets the
/// necessary environment.

static bool CreateIPythonConfig(string dest, string rootsys)
{
   string ipyconfig = dest + pathsep + ipyconfigpath;
   ofstream out(ipyconfig, ios::trunc);
   if (out.is_open()) {
      out << "import os" << endl;
      out << "rootsys = '" << rootsys << "'" << endl;
#ifdef WIN32
      out << "os.environ['PYTHONPATH']      = '%s\\lib' % rootsys + ':' + os.getenv('PYTHONPATH', '')" << endl;
      out << "os.environ['PATH']            = '%s\\bin:%s\\bin\\bin' % (rootsys,rootsys) + ':' + '%s\\lib' % rootsys + ':' + os.getenv('PATH', '')" << endl;
#else
      out << "os.environ['PYTHONPATH']      = '%s/lib' % rootsys + ':' + os.getenv('PYTHONPATH', '')" << endl;
      out << "os.environ['PATH']            = '%s/bin:%s/bin/bin' % (rootsys,rootsys) + ':' + os.getenv('PATH', '')" << endl;
      out << "os.environ['LD_LIBRARY_PATH'] = '%s/lib' % rootsys + ':' + os.getenv('LD_LIBRARY_PATH', '')" << endl;
#endif
      out.close();
      return true;
   }
   else { 
      fprintf(stderr,
              "Error installing notebook configuration files -- cannot create IPython config file at %s\n",
              ipyconfig.c_str());
      return false;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a file that stores the current commit id in it.

static bool CreateStamp(string dest)
{
   ofstream out(dest + pathsep + commitfile, ios::trunc);
   if (out.is_open()) {
      out << gROOT->GetGitCommit();
      out.close();
      return true;
   }
   else {
      fprintf(stderr,
              "Error installing notebook configuration files -- cannot create %s\n",
              commitfile.c_str());
      return false;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Spawn a Jupyter notebook customised by ROOT.

int main()
{
   // Get etc directory, it contains the ROOT notebook files to install
   string rootsys(getenv("ROOTSYS"));
#ifdef ROOTETCDIR
   string rootetc(ROOTETCDIR);
#else
   string rootetc(rootsys + pathsep + "etc");
#endif

   // If needed, install ROOT notebook files in the user's home directory
#ifdef WIN32
   string homedir(getenv("USERPROFILE"));
#else
   string homedir(getenv("HOME"));
#endif
   int inst = CheckNbInstallation(homedir);
   if (inst == -1) {
      string source(rootetc + pathsep + NB_CONF_DIR);
      string dest(homedir + pathsep + rootnbdir);
      bool res = InstallNbFiles(source, dest) &&
                 CreateIPythonConfig(dest, rootsys) &&
                 CreateStamp(dest);
      if (!res) return 1;
   }
   else if (inst == -2) return 1;

   // Set IPython directory for the ROOT notebook flavour
   string ipydir(IPYTHON_DIR_VAR + ("=" + homedir + pathsep + rootnbdir));
   putenv((char *)ipydir.c_str());

   // Execute IPython notebook
   execlp(IPYTHON_CMD, IPYTHON_CMD, NB_OPT, NULL);

   // Exec failed
   fprintf(stderr,
           "Error starting ROOT notebook -- please check that IPython notebook is installed\n");

   return 1;
}
