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

   struct FileContainerElement {
      FileContainerElement(const std::string &name, FileContainer *owner, bool isDir, size_type index)
            : fName(name), fOwner(owner), fIsDir(isDir), fIndex(index)
      {
      }
      std::string fName;
      FileContainer *fOwner;//Container-owner.
      bool fIsDir;//If entity is a directory.
      size_type fIndex;//object or directory index.
   };


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

   size_type GetNumberOfDescriptors()const;
   const FileContainerElement &GetElementDescriptor(size_type index)const;
   

   static FileContainer *CreateFileContainer(const char *fullPath);
   static void DeleteFileContainer(FileContainer *container);

private:

   void AttachPads();
   void ReadNames(const std::string &baseName, std::vector<std::string> &names)const;
   
   static void ScanDirectory(TDirectoryFile *dir, const std::set<TString> &visibleTypes, FileContainer *currentContainer);

   std::string fFileName;

   std::vector<FileContainer *>fDirectories;
   std::vector<TObject *> fObjects;
   std::vector<TString> fOptions;
   
   std::vector<Pad *> fAttachedPads;
   
   std::vector<FileContainerElement> fContentDescriptors;
   
   FileContainer &operator = (const FileContainer &rhs) = delete;
   FileContainer(const FileContainer &rhs) = delete;
};

}//namespace Browser
}//namespace iOS
}//namespace ROOT

#endif
