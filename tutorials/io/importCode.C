/// \file
/// \ingroup tutorial_io
/// \notebook -nodraw
/// Example of script showing how to create a ROOT file with subdirectories.
/// The script scans a given directory tree and recreates the same structure in the ROOT file.
/// All source files of type .h,cxx,c,dat,py are imported as TMacro objects.
/// See also the other tutorial readCode.C
/// \macro_code
///
/// \author Rene Brun

#include "TFile.h"
#include "TSystem.h"
#include "TMacro.h"

void importdir(const char *dirname) {
   char *slash = (char*)strrchr(dirname,'/');
   char *locdir;
   if (slash) locdir = slash+1;
   else       locdir = (char*)dirname;
   printf("processing dir %s\n",dirname);
   TDirectory *savdir = gDirectory;
   TDirectory *adir = savdir->mkdir(locdir);
   adir->cd();
   void *dirp = gSystem->OpenDirectory(dirname);
   if (!dirp) return;
   char *direntry;
   Long_t id, size,flags,modtime;
   //loop on all entries of this directory
   while ((direntry=(char*)gSystem->GetDirEntry(dirp))) {
      TString afile = Form("%s/%s",dirname,direntry);
      gSystem->GetPathInfo(afile,&id,&size,&flags,&modtime);
      if (direntry[0] == '.')             continue; //forget the "." and ".." special cases
      if (!strcmp(direntry,"CVS"))        continue; //forget some special directories
      if (!strcmp(direntry,"htmldoc"))    continue;
      if (strstr(dirname,"root/include")) continue;
      if (strstr(direntry,"G__"))         continue;
      if (strstr(direntry,".c")    ||
          strstr(direntry,".h")    ||
          strstr(direntry,".dat")  ||
          strstr(direntry,".py")   ||
          strstr(direntry,".C")) {
         TMacro *m = new TMacro(afile);
         m->Write(direntry);
         delete m;
      } else {
         if (flags != 3)                     continue; //must be a directory
         //we have found a valid sub-directory. Process it
         importdir(afile);
     }
  }
  gSystem->FreeDirectory(dirp);
  savdir->cd();
}
void importCode() {
   TFile *f = new TFile("code.root","recreate");
   TString dir = gROOT->GetTutorialDir();
   importdir(gSystem->UnixPathName(dir.Data())); //change the directory as you like
   delete f;
}
