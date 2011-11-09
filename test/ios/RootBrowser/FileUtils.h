#ifndef ROOT_IOSFileContainer
#define ROOT_IOSFileContainer

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// FileContainer                                                        //
//                                                                      //
// Class which owns objects read from root file.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <memory>
#include <string>
#include <set>

#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TDirectoryFile;
class TFile;

namespace ROOT {
namespace iOS {

class Pad;

namespace Browser {

enum EHistogramErrorOption {
   hetNoError,
   hetE,
   hetE1,
   hetE2,
   hetE3,
   hetE4
};


//File container inherits from TObject, to make it possible
//contain nested file containers (for TDirectoryFile, found in files).
class FileContainer : public TObject {
   
public:
   typedef std::vector<TObject *>::size_type size_type;

   FileContainer(const std::string &fileName);
   ~FileContainer();

   size_type GetNumberOfObjects()const;
   TObject *GetObject(size_type ind)const;
   const char *GetDrawOption(size_type ind)const;
   Pad *GetPadAttached(size_type ind)const;

   void SetErrorDrawOption(size_type ind, EHistogramErrorOption opt);
   EHistogramErrorOption GetErrorDrawOption(size_type ind)const;
   
   void SetMarkerDrawOption(size_type ind, bool on);
   bool GetMarkerDrawOption(size_type ind)const;

   const char *GetFileName()const;

   static FileContainer *CreateFileContainer(const char *fullPath);

private:

   static void ScanDirectory(TDirectoryFile *dir, const std::set<TString> &visibleTypes, FileContainer *currentContainer);
   void AttachPads();

   std::string fFileName;

   std::auto_ptr<TFile> fFileHandler;
   std::vector<TObject *> fFileContents;
   std::vector<TString> fOptions;
   
   std::vector<Pad *> fAttachedPads;
};

//This is the function to be called from Obj-C++ code.
//Return: non-null pointer in case file was
//opened and its content read.
FileContainer *CreateFileContainer(const char *fileName);
//Just for symmetry.
void DeleteFileContainer(FileContainer *container);

}//namespace Browser
}//namespace iOS
}//namespace ROOT

#endif
