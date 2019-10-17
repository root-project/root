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


#include "ROOT/RFileBrowsable.hxx"

#include "ROOT/RLogger.hxx"

#include "TSystem.h"
#include "TKey.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TFile.h"

#include <sstream>
#include <fstream>

using namespace std::string_literals;

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Browsable;

/** \class ROOT::Experimental::RSysDirLevelIter
\ingroup rbrowser

Iterator over single file level
*/


class RSysDirLevelIter : public RLevelIter {
   std::string fPath;        ///<! fully qualified path
   void *fDir{nullptr};      ///<! current directory handle
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

         // one have to use const char* to correctly check for nullptr
         const char *name = gSystem->GetDirEntry(fDir);

         if (!name) {
            CloseDir();
            return false;
         }

         std::string sname = name;

         if ((sname == ".") || (sname == ".."))
            continue;

         TestDirEntry(sname);
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
   explicit RSysDirLevelIter(const std::string &path = "") : fPath(path) { OpenDir(); }

   virtual ~RSysDirLevelIter() { CloseDir(); }

   bool Reset() override { return OpenDir(); }

   bool Next() override { return NextDirEntry(); }

   bool Find(const std::string &name) override { return FindDirEntry(name); }

   bool HasItem() const override { return !fCurrentName.empty(); }

   std::string GetName() const override { return fCurrentName; }

   /** Returns true if item can have childs and one should try to create iterator (optional) */
   int CanHaveChilds() const override
   {
      if (R_ISDIR(fCurrentStat.fMode))
         return 1;

      if ((fCurrentName.length() > 5) && (fCurrentName.rfind(".root") == fCurrentName.length() - 5))
         return 1;

      return 0;
   }

   static std::string GetFileIcon(const std::string &fname);

   std::unique_ptr<RBrowserItem> CreateBrowserItem() override
   {
      auto item = std::make_unique<RBrowserFileItem>(GetName(), CanHaveChilds());

      // this is construction of current item
      char tmp[256];
      Long64_t _fsize, bsize;

      item->type     = fCurrentStat.fMode;
      item->size     = fCurrentStat.fSize;
      item->uid      = fCurrentStat.fUid;
      item->gid      = fCurrentStat.fGid;
      item->modtime  = fCurrentStat.fMtime;
      item->islink   = fCurrentStat.fIsLink;
      item->isdir    = R_ISDIR(fCurrentStat.fMode);

      if (item->isdir)
         item->SetIcon("sap-icon://folder-blank"s);
      else
         item->SetIcon(GetFileIcon(GetName()));

      // file size
      _fsize = bsize = item->size;
      if (_fsize > 1024) {
         _fsize /= 1024;
         if (_fsize > 1024) {
            // 3.7MB is more informative than just 3MB
            snprintf(tmp, sizeof(tmp), "%lld.%lldM", _fsize/1024, (_fsize%1024)/103);
         } else {
            snprintf(tmp, sizeof(tmp), "%lld.%lldK", bsize/1024, (bsize%1024)/103);
         }
      } else {
         snprintf(tmp, sizeof(tmp), "%lld", bsize);
      }
      item->fsize = tmp;

      // modification time
      time_t loctime = (time_t) item->modtime;
      struct tm *newtime = localtime(&loctime);
      if (newtime) {
         snprintf(tmp, sizeof(tmp), "%d-%02d-%02d %02d:%02d", newtime->tm_year + 1900,
                  newtime->tm_mon+1, newtime->tm_mday, newtime->tm_hour,
                  newtime->tm_min);
         item->mtime = tmp;
      } else {
         item->mtime = "1901-01-01 00:00";
      }

      // file type
      snprintf(tmp, sizeof(tmp), "%c%c%c%c%c%c%c%c%c%c",
               (item->islink ?
                'l' :
                R_ISREG(item->type) ?
                '-' :
                (R_ISDIR(item->type) ?
                 'd' :
                 (R_ISCHR(item->type) ?
                  'c' :
                  (R_ISBLK(item->type) ?
                   'b' :
                   (R_ISFIFO(item->type) ?
                    'p' :
                    (R_ISSOCK(item->type) ?
                     's' : '?' )))))),
               ((item->type & kS_IRUSR) ? 'r' : '-'),
               ((item->type & kS_IWUSR) ? 'w' : '-'),
               ((item->type & kS_ISUID) ? 's' : ((item->type & kS_IXUSR) ? 'x' : '-')),
               ((item->type & kS_IRGRP) ? 'r' : '-'),
               ((item->type & kS_IWGRP) ? 'w' : '-'),
               ((item->type & kS_ISGID) ? 's' : ((item->type & kS_IXGRP) ? 'x' : '-')),
               ((item->type & kS_IROTH) ? 'r' : '-'),
               ((item->type & kS_IWOTH) ? 'w' : '-'),
               ((item->type & kS_ISVTX) ? 't' : ((item->type & kS_IXOTH) ? 'x' : '-')));
      item->ftype = tmp;

      struct UserGroup_t *user_group = gSystem->GetUserInfo(item->uid);
      if (user_group) {
         item->fuid = user_group->fUser;
         item->fgid = user_group->fGroup;
         delete user_group;
      } else {
         item->fuid = std::to_string(item->uid);
         item->fgid = std::to_string(item->gid);
      }

      return item;
   }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override
   {
      if (!R_ISDIR(fCurrentStat.fMode) && (fCurrentName.length() > 5) && (fCurrentName.rfind(".root") == fCurrentName.length()-5))
         return std::make_shared<TDirectoryElement>(fCurrentName);

      return std::make_shared<SysFileElement>(fCurrentStat, fPath, fCurrentName);
   }

};



/////////////////////////////////////////////////////////////////////////////////
/// Get icon for the type of given file name

std::string RSysDirLevelIter::GetFileIcon(const std::string &fname)
{
   auto EndsWith = [fname](const std::string &suffix) {
      return (fname.length() > suffix.length()) ? (0 == fname.compare (fname.length() - suffix.length(), suffix.length(), suffix)) : false;
   };

   if ((EndsWith(".c")) ||
       (EndsWith(".cpp")) ||
       (EndsWith(".cxx")) ||
       (EndsWith(".c++")) ||
       (EndsWith(".cxx")) ||
       (EndsWith(".h")) ||
       (EndsWith(".hpp")) ||
       (EndsWith(".hxx")) ||
       (EndsWith(".h++")) ||
       (EndsWith(".py")) ||
       (EndsWith(".txt")) ||
       (EndsWith(".cmake")) ||
       (EndsWith(".dat")) ||
       (EndsWith(".log")) ||
       (EndsWith(".js")))
      return "sap-icon://document-text"s;
   if ((EndsWith(".bmp")) ||
       (EndsWith(".gif")) ||
       (EndsWith(".jpg")) ||
       (EndsWith(".png")) ||
       (EndsWith(".svg")))
      return "sap-icon://picture"s;
  if (EndsWith(".root"))
      return "sap-icon://org-chart"s;

   return "sap-icon://document"s;
}




SysFileElement::SysFileElement(const std::string &filename) : fFileName(filename)
{
   if (gSystem->GetPathInfo(fFileName.c_str(), fStat)) {
      if (fStat.fIsLink) {
         R__ERROR_HERE("Browserv7") << "Broken symlink of " << fFileName;
      } else {
         R__ERROR_HERE("Browserv7") << "Can't read file attributes of \"" << fFileName
                                    << "\" err:" << gSystem->GetError();
      }
   }
}

std::string SysFileElement::GetFullName() const
{
   std::string path = fDirName;
   if (!path.empty() && (path.rfind("/") != path.length() - 1))
      path.append("/");
   path.append(fFileName);
   return path;
}


std::unique_ptr<RLevelIter> SysFileElement::GetChildsIter()
{
   if (!R_ISDIR(fStat.fMode))
      return nullptr;

   // TODO: support .root file and all other file types later

   return std::make_unique<RSysDirLevelIter>(GetFullName());
}

bool SysFileElement::HasTextContent() const
{
   return RSysDirLevelIter::GetFileIcon(GetName()) == "sap-icon://document-text"s;
}

std::string SysFileElement::GetTextContent()
{
   std::ifstream t(GetFullName());
   return std::string(std::istreambuf_iterator<char>(t), std::istreambuf_iterator<char>());
}


// ===============================================================================================================


class TDirectoryLevelIter : public RLevelIter {
   TDirectory *fDir{nullptr};         ///<! current directory handle
   std::unique_ptr<TIterator>  fIter; ///<! created iterator
   TKey *fKey{nullptr};               ///<! currently selected key
   std::string fCurrentName;          ///<! current key name

   bool CreateIter()
   {
      if (!fDir) return false;
      fIter.reset(fDir->GetListOfKeys()->MakeIterator());
      fKey = nullptr;
      return true;
   }

   void CloseIter()
   {
      fIter.reset(nullptr);
      fKey = nullptr;
   }

   bool NextDirEntry()
   {
      fCurrentName.clear();
      if (!fIter) return false;

      fKey = dynamic_cast<TKey *>(fIter->Next());

      if (!fKey) {
         fIter.reset();
         return false;
      }

      fCurrentName = fKey->GetName();
      fCurrentName.append(";");
      fCurrentName.append(std::to_string(fKey->GetCycle()));

      return true;
   }

public:
   explicit TDirectoryLevelIter(TDirectory *dir) : fDir(dir) { CreateIter(); }

   virtual ~TDirectoryLevelIter() = default;

   bool Reset() override { return CreateIter(); }

   bool Next() override { return NextDirEntry(); }

   // use default implementation for now
   // bool Find(const std::string &name) override { return FindDirEntry(name); }

   bool HasItem() const override { return fKey && !fCurrentName.empty(); }

   std::string GetName() const override { return fCurrentName; }

   int CanHaveChilds() const override
   {
      std::string clname = fKey->GetClassName();
      return (clname.find("TDirectory") == 0) ? 1 : 0;
   }

   std::string GetClassIcon(const std::string &classname)
   {
      if (classname == "TTree" || classname == "TNtuple")
         return "sap-icon://tree"s;
      if (classname == "TDirectory" || classname == "TDirectoryFile")
         return "sap-icon://folder-blank"s;

      return "sap-icon://electronic-medical-record"s;
   }

   /** Create element for the browser */
   std::unique_ptr<RBrowserItem> CreateBrowserItem() override
   {
      auto item = std::make_unique<RBrowserTKeyItem>(GetName(), CanHaveChilds());

      item->className = fKey->GetClassName();

      item->SetIcon(GetClassIcon(item->className));

      return item;
   }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override;

};

// ===============================================================================================================



class TKeyElement : public RElement {
   TDirectory *fDir{nullptr};
   TKey *fKey{nullptr};

public:
   TKeyElement(TDirectory *dir, TKey *key) : fDir(dir), fKey(key) {}

   virtual ~TKeyElement() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override
   {
      std::string name = fKey->GetName();
      name.append(";");
      name.append(std::to_string(fKey->GetCycle()));
      return name;
   }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override { return fKey->GetTitle(); }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override
   {
      std::string clname = fKey->GetClassName();

      if (clname.find("TDirectory") != 0)
         return nullptr;

      auto subdir = fDir->GetDirectory(GetName().c_str());
      if (subdir)
         return std::make_unique<TDirectoryLevelIter>(subdir);

      return nullptr;
   }

   /** Temporary solution, later better interface should be provided */
   TObject *GetObjectToDraw() override
   {
      std::string clname = fKey->GetClassName();

      // TODO: this check has to be performed in RBrowsable
      if ((clname.find("TTree") == 0) || (clname.find("TChain") == 0) || (clname.find("TDirectory") == 0))
         return nullptr;


      return fKey->ReadObj();
   }

};

// ==============================================================================================


std::shared_ptr<RElement> TDirectoryLevelIter::GetElement()
{
   return std::make_shared<TKeyElement>(fDir, fKey);
}

// ==============================================================================================

TDirectoryElement::TDirectoryElement(const std::string &fname, TDirectory *dir)
{
   fFileName = fname;
   fDir = dir;
}

TDirectoryElement::~TDirectoryElement()
{
}

///////////////////////////////////////////////////////////////////
/// Get TDirectory. Checks if parent file is still there. If not, means it was closed outside ROOT

TDirectory *TDirectoryElement::GetDir() const
{
   if (fDir) {
      if (!gROOT->GetListOfFiles()->FindObject(fDir->GetFile()))
         const_cast<TDirectoryElement *>(this)->fDir = nullptr;
   } else if (!fFileName.empty()) {
      const_cast<TDirectoryElement *>(this)->fDir = TFile::Open(fFileName.c_str());
   }

   return fDir;
}


/** Name of RBrowsable, must be provided in derived classes */
std::string TDirectoryElement::GetName() const
{
   if (fDir)
      return fDir->GetName();

   if (!fFileName.empty()) {
      auto pos = fFileName.rfind("/");
      return ((pos == std::string::npos) || (pos > fFileName.length()-2)) ? fFileName : fFileName.substr(pos+1);
   }

   return ""s;
}

/** Title of RBrowsable (optional) */
std::string TDirectoryElement::GetTitle() const
{
   auto dir = GetDir();

   return dir ? dir->GetTitle() : "";
}

std::unique_ptr<RLevelIter> TDirectoryElement::GetChildsIter()
{
   auto dir = GetDir();

   return dir ? std::make_unique<TDirectoryLevelIter>(dir) : nullptr;
}


