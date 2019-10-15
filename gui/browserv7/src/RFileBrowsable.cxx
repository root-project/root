/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/// \file ROOT/RFileBrowsable.cxx
/// \ingroup rbrowser
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2019-10-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!


#include "ROOT/RBrowsable.hxx"

#include "ROOT/RLogger.hxx"

#include "TSystem.h"

using namespace ROOT::Experimental;



class RFileInfo : public RBrowsableInfo {
   FileStat_t fStat;       ///<! file stat object
   std::string fDirName;   ///<! fully-qualified directory name
   std::string fFileName;  ///<! file name in current dir

   std::string GetFullName() const
   {
      std::string path = fDirName;
      if (!path.empty() && (path.rfind("/") != path.length()-1))
         path.append("/");
      path.append(fFileName);
      return path;
   }

public:
   RFileInfo(const std::string &filename) : fFileName(filename)
   {
      if (gSystem->GetPathInfo(fFileName.c_str(), fStat)) {
          if (fStat.fIsLink) {
             R__ERROR_HERE("Browserv7") << "Broken symlink of " << fFileName;
          } else {
             R__ERROR_HERE("Browserv7") << "Can't read file attributes of \"" << fFileName << "\" err:" << gSystem->GetError();
          }
      }
   }

   RFileInfo(const FileStat_t& stat, const std::string &dirname, const std::string &filename) : fStat(stat), fDirName(dirname), fFileName(filename)
   {
   }

   virtual ~RFileInfo() = default;

   /** Class information for file not provided */
   const TClass *GetGlass() const override { return nullptr; }

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override { return fFileName; }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override { return GetFullName(); }

   /** Returns true if item can have childs and one should try to create iterator (optional) */
   int CanHaveChilds() const override
   {
      if (R_ISDIR(fStat.fMode)) return 1;

      if (fFileName.rfind(".root") == fFileName.length()-5)
         return 1;

      return 0;
   }

   std::unique_ptr<RBrowsableLevelIter> GetChildsIter() override;
};




/** \class ROOT::Experimental::RDirectoryLevelIter
\ingroup rbrowser

Iterator over single file level
*/


class RDirectoryLevelIter : public RBrowsableLevelIter {
   std::string fPath;   ///<! fully qualified path
   void *fDir{nullptr}; ///<! current directory handle
   std::string fCurrentName; ///<! current file name
   FileStat_t fCurrentStat;  ///<! stat for current file name

   bool OpenDir()
   {
      if (fDir)
         CloseDir();

      fDir = gSystem->OpenDirectory(fPath.c_str());
      if (!fDir) {
         R__ERROR_HERE("Browserv7") << "Fail to open directory " << fPath;
         return false;
      }

      return true;
   }

   void CloseDir()
   {
      if (fDir)
         gSystem->FreeDirectory(fDir);
      fDir = nullptr;
      fCurrentName.clear();
   }

   bool TestDirEntry(const std::string &name)
   {
      std::string path = fPath;
      if (path.rfind("/") != path.length()-1)
         path.append("/");
      path.append(name);

      if (gSystem->GetPathInfo(path.c_str(), fCurrentStat)) {
         if (fCurrentStat.fIsLink) {
            R__ERROR_HERE("Browserv7") << "Broken symlink of " << path;
         } else {
            R__ERROR_HERE("Browserv7") << "Can't read file attributes of \"" <<  path << "\" err:" << gSystem->GetError();
         }
         return false;
      }

      fCurrentName = name;
      return true;
   }

   bool NextDirEntry()
   {
      fCurrentName.clear();

      if (!fDir)
         return false;

      while (fCurrentName.empty()) {
         std::string name = gSystem->GetDirEntry(fDir);

        if (name.empty()) {
           CloseDir();
           return false;
        }

        if ((name == ".") || (name == ".."))
           continue;

        TestDirEntry(name);
      }


      return true;
   }

   bool FindDirEntry(const std::string &name)
   {
      if (!fDir && !OpenDir())
         return false;

      return TestDirEntry(name);
   }

public:
   explicit RDirectoryLevelIter(const std::string &path = "") : fPath(path) {}

   virtual ~RDirectoryLevelIter() { CloseDir(); }

   bool Reset() override { return OpenDir(); }

   bool Next() override { return NextDirEntry(); }

   bool Find(const std::string &name) override { return FindDirEntry(name); }

   bool HasItem() const override { return !fCurrentName.empty(); }

   std::string GetName() const override { return fCurrentName; }

   /** Returns full information for current element */
   std::unique_ptr<RBrowsableInfo> GetInfo() override
   {
      return std::make_unique<RFileInfo>(fCurrentStat, fPath, fCurrentName);
   }
};




std::unique_ptr<RBrowsableLevelIter> RFileInfo::GetChildsIter()
{
   if (!R_ISDIR(fStat.fMode))
      return nullptr;

   // TODO: support .root file and all other file types later

   return std::make_unique<RDirectoryLevelIter>(GetFullName());
}


