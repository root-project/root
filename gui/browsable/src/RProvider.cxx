/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RProvider.hxx>

#include <ROOT/RLogger.hxx>

using namespace ROOT::Experimental::Browsable;
using namespace std::string_literals;

//////////////////////////////////////////////////////////////////////////////////
// Provide map of browsing for different classes

RProvider::BrowseMap_t &RProvider::GetBrowseMap()
{
   static RProvider::BrowseMap_t sMap;
   return sMap;
}

//////////////////////////////////////////////////////////////////////////////////
// Provide map of files opening

RProvider::FileMap_t &RProvider::GetFileMap()
{
   static RProvider::FileMap_t sMap;
   return sMap;
}


//////////////////////////////////////////////////////////////////////////////////
// Destructor
/// Automatically unregister provider from global lists

RProvider::~RProvider()
{
   // here to remove all correspondent entries
   auto &fmap = GetFileMap();

   for (auto fiter = fmap.begin();fiter != fmap.end();) {
      if (fiter->second.provider == this)
         fiter = fmap.erase(fiter);
      else
         fiter++;
   }

   auto &bmap = GetBrowseMap();
   for (auto biter = bmap.begin(); biter != bmap.end();) {
      if (biter->second.provider == this)
         biter = bmap.erase(biter);
      else
         biter++;
   }
}

//////////////////////////////////////////////////////////////////////////////////
// Register file open function for specified extension

void RProvider::RegisterFile(const std::string &extension, FileFunc_t func)
{
    auto &fmap = GetFileMap();

    if ((extension != "*") && (fmap.find(extension) != fmap.end()))
       R__ERROR_HERE("Browserv7") << "Provider for file extension  " << extension << " already exists";

    fmap.emplace(extension, StructFile{this,func});
}

//////////////////////////////////////////////////////////////////////////////////
// Register browse function for specified class

void RProvider::RegisterBrowse(const TClass *cl, BrowseFunc_t func)
{
    auto &bmap = GetBrowseMap();

    if (cl && (bmap.find(cl) != bmap.end()))
       R__ERROR_HERE("Browserv7") << "Browse provider for class " << cl->GetName() << " already exists";

    bmap.emplace(cl, StructBrowse{this,func});
}

//////////////////////////////////////////////////////////////////////////////////
// remove provider from all registered lists

std::shared_ptr<RElement> RProvider::OpenFile(const std::string &extension, const std::string &fullname)
{
   auto &fmap = GetFileMap();

   auto iter = fmap.find(extension);

   if (iter != fmap.end()) {
      auto res = iter->second.func(fullname);
      if (res) return res;
   }

   for (auto &pair : fmap)
      if ((pair.first == "*") || (pair.first == extension)) {
         auto res = pair.second.func(fullname);
         if (res) return res;
      }

   return nullptr;
}

/////////////////////////////////////////////////////////////////////////
/// Create browsable element for the object
/// Created element may take ownership over the object

std::shared_ptr<RElement> RProvider::Browse(std::unique_ptr<RHolder> &object)
{
   if (!object)
      return nullptr;

   auto &bmap = GetBrowseMap();

   auto cl = object->GetClass();
   if (!cl)
      return nullptr;

   auto iter = bmap.find(cl);

   if (iter != bmap.end()) {
      auto res = iter->second.func(object);
      if (res || !object) return res;
   }

   for (auto &pair : bmap)
      if ((pair.first == nullptr) || (cl == pair.first)) {
         auto res = pair.second.func(object);
         if (res || !object) return res;
      }

   return nullptr;
}

/////////////////////////////////////////////////////////////////////
/// Return icon name for the given class
/// TODO: should be factorized out from here

std::string RProvider::GetClassIcon(const std::string &classname)
{
   if (classname == "TTree" || classname == "TNtuple")
      return "sap-icon://tree"s;
   else if (classname == "TDirectory" || classname == "TDirectoryFile")
      return "sap-icon://folder-blank"s;
   else if (classname.find("TLeaf") == 0)
      return "sap-icon://e-care"s;

   return "sap-icon://electronic-medical-record"s;
}
