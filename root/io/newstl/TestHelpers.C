#include "TestHelpers.h"
#include "TestOutput.h"
#include "TSystem.h"
#include "TList.h"
#include "TRegexp.h"
#include "TObjString.h"

void fillListOfDir(TList &l) {
   
   TString directory = ".";
   void *dir = gSystem->OpenDirectory(directory);

   const char *file = 0;
   if (dir) {

      //create a TList to store the file names (not yet sorted)
      TString basename = ".-..-..";
      TRegexp re(basename,kFALSE);

      while ((file = gSystem->GetDirEntry(dir))) {
         if (!strcmp(file,".") || !strcmp(file,"..")) continue;
         TString s = file;
//          cout << "found the directory " << file << endl;
         if ( (basename!=file) && s.Index(re) == kNPOS) continue;

         TString vfile = gSystem->ConcatFileName(file,"vector.root");
         if (gSystem->GetPathInfo(vfile,(Long_t*)0,(Long_t*)0,(Long_t*)0,0)==0) {
//             cout << "found vector in " << file << endl;
            l.Add(new TObjString(file));
         } else {
//             cout << "did not find vector in " << file << endl;
         }

      }
      gSystem->FreeDirectory(dir);

      //sort the files in alphanumeric order
      l.Sort();

      TIter next(&l);
      TObjString *obj;
      while ((obj = (TObjString*)next())) {
         file = obj->GetName();
//          cout << "found the directory " << obj->GetName() << endl;
      }
   }
}
#ifdef __MAKECINT__
#pragma link C++ function DebugTest;
#pragma link C++ class pair<float,int>+;
#pragma link C++ class pair<std::string,double>+;
#pragma link C++ class GHelper<float>+;
#pragma link C++ class GHelper<GHelper<float> >+;
#pragma link C++ class GHelper<GHelper<GHelper<float> > >+;
#endif
