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


class FileContainer {
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
   TObject *GetObject(size_type ind)const;
   const char *GetDrawOption(size_type ind)const;
   Pad *GetPadAttached(size_type ind)const;
   void SetErrorDrawOption(size_type ind, EHistogramErrorOption opt);
   EHistogramErrorOption GetErrorDrawOption(size_type ind)const;
      
   void SetMarkerDrawOption(size_type ind, bool on);
   bool GetMarkerDrawOption(size_type ind)const;

   size_type GetNumberOfDirectories()const;   
   FileContainer *GetDirectory(size_type ind)const;

   const char *GetFileName()const;

   //Names for nested objects and directories.
   size_type ReadNames()const;
   const std::string &GetName(size_type ind)const;

   //Search for object or directory, return number of paths to this
   //entity found. If this FileContainer has such an object in fObjects,
   //the path is 'this' pointer.
   size_type FindObject(const std::string &objectName)const;
   const std::vector<const FileContainer *> &GetPath(size_type pathIndex)const;
   //These are the functions to be called from Obj-C++ code.
   //Return: non-null pointer in case file was
   //opened and its content read.

   static FileContainer *CreateFileContainer(const char *fullPath);
   static void DeleteFileContainer(FileContainer *container);

private:

   void AttachPads();
   
   static void ScanDirectory(TDirectoryFile *dir, const std::set<TString> &visibleTypes, FileContainer *currentContainer);


   std::string fFileName;

   std::vector<FileContainer *>fDirectories;
   std::vector<TObject *> fObjects;
   std::vector<TString> fOptions;
   
   std::vector<Pad *> fAttachedPads;
   
   mutable std::vector<std::vector<const FileContainer *>> fSearchPaths;
   
   //Actually, I can use vector of const char *, which will point to
   //real allocated strings. But just not to care about these strings life-time ...
   mutable std::vector<std::string> fNames;
   
   FileContainer &operator = (const FileContainer &rhs) = delete;
   FileContainer(const FileContainer &rhs) = delete;
};

}//namespace Browser
}//namespace iOS
}//namespace ROOT

#endif
