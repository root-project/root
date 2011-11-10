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
   //Auto ptr must delete file container in case of exception
   //in CreateFileContainer and so needs an access to private dtor.
   friend class std::auto_ptr<FileContainer>;
public:
   typedef std::vector<TObject *>::size_type size_type;

private:
   FileContainer(const std::string &fileName);
   ~FileContainer();

public:
   size_type GetNumberOfObjects()const;
   size_type GetNumberOfNondirObjects()const;
   TObject *GetObject(size_type ind)const;
   const char *GetDrawOption(size_type ind)const;
   Pad *GetPadAttached(size_type ind)const;

   void SetErrorDrawOption(size_type ind, EHistogramErrorOption opt);
   EHistogramErrorOption GetErrorDrawOption(size_type ind)const;
   
   void SetMarkerDrawOption(size_type ind, bool on);
   bool GetMarkerDrawOption(size_type ind)const;

   const char *GetFileName()const;

   //These are the function to be called from Obj-C++ code.
   //Return: non-null pointer in case file was
   //opened and its content read.

   static FileContainer *CreateFileContainer(const char *fullPath);
   static void DeleteFileContainer(FileContainer *container);

private:

   static void ScanDirectory(TDirectoryFile *dir, const std::set<TString> &visibleTypes, FileContainer *currentContainer);
   void AttachPads();

   std::string fFileName;
   size_type fNondirObjects;

   std::vector<TObject *> fFileContents;
   std::vector<TString> fOptions;
   
   std::vector<Pad *> fAttachedPads;
   
   FileContainer &operator = (const FileContainer &rhs) = delete;
   FileContainer(const FileContainer &rhs) = delete;
};

}//namespace Browser
}//namespace iOS
}//namespace ROOT

#endif
