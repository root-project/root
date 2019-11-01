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


#include "ROOT/RBrowsableSysFile.hxx"

#include "ROOT/RLogger.hxx"

#include "TSystem.h"
#include "TROOT.h"
#include "TBase64.h"

#include <sstream>
#include <fstream>

using namespace std::string_literals;

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Browsable;

/** \class RSysDirLevelIter
\ingroup rbrowser

Iterator over files in in sub-directory
*/

class RSysDirLevelIter : public RLevelIter {
   std::string fPath;        ///<! fully qualified path
   void *fDir{nullptr};      ///<! current directory handle
   std::string fCurrentName; ///<! current file name
   FileStat_t fCurrentStat;  ///<! stat for current file name

   /** Open directory for listing */
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

   /** Close directory for listing */
   void CloseDir()
   {
      if (fDir)
         gSystem->FreeDirectory(fDir);
      fDir = nullptr;
      fCurrentName.clear();
   }

   /** Check if entry of that name exists */
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

   /** Trying to gen next entry */
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

   /** Try to find file directly by name */
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
      Long64_t _fsize = item->size, bsize = item->size;
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
      if (!R_ISDIR(fCurrentStat.fMode) && (fCurrentName.length() > 5) && (fCurrentName.rfind(".root") == fCurrentName.length()-5)) {
         std::string fullname = fPath;
         if (!fullname.empty()) fullname.append("/");
         fullname.append(fCurrentName);
         auto elem = RProvider::OpenFile("root", fullname);
         if (elem) return elem;
      }

      return std::make_shared<SysFileElement>(fCurrentStat, fPath, fCurrentName);
   }

};


/////////////////////////////////////////////////////////////////////////////////
/// Get icon for the type of given file name

std::string RSysDirLevelIter::GetFileIcon(const std::string &fname)
{
   auto EndsWith = [fname](const std::string &suffix) {
      std::string name = fname;
      std::transform(name.begin(), name.end(), name.begin(), ::tolower);
      return (name.length() > suffix.length()) ? (0 == name.compare (name.length() - suffix.length(), suffix.length(), suffix)) : false;
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
       (EndsWith(".xml")) ||
       (EndsWith(".htm")) ||
       (EndsWith(".html")) ||
       (EndsWith(".json")) ||
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


/////////////////////////////////////////////////////////////////////////////////
/// Create file element

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

/////////////////////////////////////////////////////////////////////////////////
/// Returns full file name - including directory

std::string SysFileElement::GetFullName() const
{
   std::string path = fDirName;
   if (!path.empty() && (path.rfind("/") != path.length() - 1))
      path.append("/");
   path.append(fFileName);
   return path;
}

/////////////////////////////////////////////////////////////////////////////////
/// Returns iterator for files in directory

std::unique_ptr<RLevelIter> SysFileElement::GetChildsIter()
{
   if (!R_ISDIR(fStat.fMode))
      return nullptr;

   return std::make_unique<RSysDirLevelIter>(GetFullName());
}

/////////////////////////////////////////////////////////////////////////////////
/// Returns file content of requested kind

std::string SysFileElement::GetContent(const std::string &kind)
{
   if ((GetContentKind(kind) == kText) && (RSysDirLevelIter::GetFileIcon(GetName()) == "sap-icon://document-text"s)) {
      std::ifstream t(GetFullName());
      return std::string(std::istreambuf_iterator<char>(t), std::istreambuf_iterator<char>());
   }

   if ((GetContentKind(kind) == kImage) && (RSysDirLevelIter::GetFileIcon(GetName()) == "sap-icon://picture"s)) {
      std::ifstream t(GetFullName());
      std::string content = std::string(std::istreambuf_iterator<char>(t), std::istreambuf_iterator<char>());

      auto encode = TBase64::Encode(content.data(), content.length());

      auto pos = GetName().rfind(".");

      return "data:image/"s  + GetName().substr(pos+1) + ";base64,"s + encode.Data();
   }

   return ""s;
}
