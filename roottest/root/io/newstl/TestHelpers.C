#include "TestHelpers.h"
#include "TestOutput.h"
#include "TSystem.h"
#include "TList.h"
#include "TRegexp.h"
#include "TObjString.h"
#include <utility>

void fillListOfDir(const TString &directory, TList &l) {

   void *dir = gSystem->OpenDirectory(directory);

   const char *file = 0;
   if (dir) {

      //create a TList to store the file names (not yet sorted)
      TString basename = ".-..-..";
      TRegexp re(basename,kFALSE);

      while ((file = gSystem->GetDirEntry(dir))) {
         if (!strcmp(file,".") || !strcmp(file,"..")) continue;
         // Skip 'latest' as it is a symlink
         if (strcmp(file,"latest")==0) continue;

         TString s = file;
         if ( (basename!=file) && s.Index(re) == kNPOS) continue;

         TString dirname = file;
         if (directory != ".") {
            auto _dirname = gSystem->ConcatFileName(directory, file);
            dirname = _dirname;
            delete [] _dirname;
         }

         auto _vfile = gSystem->ConcatFileName(dirname, "vector.root");
         TString vfile = _vfile;
         delete [] _vfile;

         if (gSystem->GetPathInfo(vfile,(Long_t*)0,(Long_t*)0,(Long_t*)0,0)==0) {
            l.Add(new TObjString(dirname));
         } else {
//             cout << "did not find vector in " << file << endl;
         }

      }
      gSystem->FreeDirectory(dir);

   }
}

void fillListOfDir(TList &l) {
   fillListOfDir(".", l);
   fillListOfDir("ArchivedDataFiles", l);
   const char *otherdir = gSystem->Getenv("ROOT_NEWSTL_DATAFILE_DIR");
   if (otherdir && otherdir[0])
      fillListOfDir(otherdir, l);

   // Sort the files in alphanumeric order
   l.Sort();

   if (gDebug > 3) {
      TIter next(&l);
      TObjString *obj;
      while ((obj = (TObjString*)next())) {
         const char *file = obj->GetName();
         cout << "found the directory " << file << endl;
      }
   }
}

#ifdef __MAKECINT__
#pragma link C++ function DebugTest;
//#pragma link C++ class pair<float,int>+;
//#pragma link C++ class pair<std::string,double>+;'
//#pragma create TClass pair<std::string,double>+;
#pragma link C++ class GHelper<float>+;
#pragma link C++ class GHelper<GHelper<float> >+;
#pragma link C++ class GHelper<GHelper<GHelper<float> > >+;
#endif
