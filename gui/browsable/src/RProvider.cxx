/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RProvider.hxx>

#include <ROOT/RLogger.hxx>

#include "TBaseClass.h"
#include "TList.h"
#include "TSystem.h"

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
// Returns map of registered drawing functions for v6 canvas

RProvider::Draw6Map_t &RProvider::GetDraw6Map()
{
   static RProvider::Draw6Map_t sMap;
   return sMap;
}

//////////////////////////////////////////////////////////////////////////////////
// Returns map of registered drawing functions for v7 canvas

RProvider::Draw7Map_t &RProvider::GetDraw7Map()
{
   static RProvider::Draw7Map_t sMap;
   return sMap;
}

//////////////////////////////////////////////////////////////////////////////////
// Destructor
/// Automatically unregister provider from all maps

RProvider::~RProvider()
{
   // here to remove all correspondent entries
   CleanThis(GetFileMap());

   CleanThis(GetBrowseMap());

   CleanThis(GetDraw6Map());

   CleanThis(GetDraw7Map());
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
// Register drawing function for v6 canvas

void RProvider::RegisterDraw6(const TClass *cl, Draw6Func_t func)
{
    auto &bmap = GetDraw6Map();

    if (cl && (bmap.find(cl) != bmap.end()))
       R__ERROR_HERE("Browserv7") << "Draw v6 handler for class " << cl->GetName() << " already exists";

    bmap.emplace(cl, StructDraw6{this, func});
}

//////////////////////////////////////////////////////////////////////////////////
// Register drawing function for v7 canvas

void RProvider::RegisterDraw7(const TClass *cl, Draw7Func_t func)
{
    auto &bmap = GetDraw7Map();

    if (cl && (bmap.find(cl) != bmap.end()))
       R__ERROR_HERE("Browserv7") << "Draw v7 handler for class " << cl->GetName() << " already exists";

    bmap.emplace(cl, StructDraw7{this, func});
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


template<class Map_t, class Iterator_t>
bool ScanProviderMap(Map_t &fmap, const TClass *cl, bool test_all, std::function<bool(Iterator_t &)> check_func)
{
   if (!cl)
      return false;

   TClass *testcl = const_cast<TClass *>(cl);
   while (testcl) {
      auto iter = fmap.find(testcl);
      if (iter != fmap.end())
         if (check_func(iter))
            return true;

      auto bases = testcl->GetListOfBases();

      testcl = bases && (bases->GetSize() > 0) ? dynamic_cast<TBaseClass *>(bases->First())->GetClassPointer() : nullptr;
   }

   if (test_all) {
      auto iter = fmap.begin();
      while (iter != fmap.end()) {
         if (!iter->first && check_func(iter))
            return true;
         iter++;
      }
   }

   return false;
}


/////////////////////////////////////////////////////////////////////////
/// Create browsable element for the object
/// Created element may take ownership over the object

std::shared_ptr<RElement> RProvider::Browse(std::unique_ptr<RHolder> &object)
{
   std::shared_ptr<RElement> res;

   if (object)
      ScanProviderMap<BrowseMap_t,BrowseMap_t::iterator>(GetBrowseMap(), object->GetClass(), true,
            [&res, &object] (BrowseMap_t::iterator &iter) -> bool {
              res = iter->second.func(object);
              return (res || !object) ? true : false;
            }
      );

   return res;
}

/////////////////////////////////////////////////////////////////////////////////
/// Invoke drawing of object on TCanvas sub-pad
/// All existing providers are checked, first checked are class matches (including direct parents)

bool RProvider::Draw6(TVirtualPad *subpad, std::unique_ptr<Browsable::RHolder> &object, const std::string &opt)
{
   if (!object || !object->GetClass())
      return false;

   auto draw_func = [subpad, &object, &opt](Draw6Map_t::iterator &iter) -> bool {
      return iter->second.func(subpad, object, opt);
   };

   if (ScanProviderMap<Draw6Map_t, Draw6Map_t::iterator>(GetDraw6Map(), object->GetClass(), false, draw_func))
      return true;

   if (object->GetClass()->InheritsFrom("TLeaf"))
      gSystem->Load("libROOTTreeDrawProvider");
   else if (object->GetClass()->InheritsFrom(TObject::Class()))
      gSystem->Load("libROOTObjectDrawProvider");
   else
      return false;

   return ScanProviderMap<Draw6Map_t, Draw6Map_t::iterator>(GetDraw6Map(), object->GetClass(), true, draw_func);
}

/////////////////////////////////////////////////////////////////////////////////
/// Invoke drawing of object on RCanvas sub-pad
/// All existing providers are checked, first checked are class matches (including direct parents)

bool RProvider::Draw7(std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RHolder> &object, const std::string &opt)
{
   if (!object || !object->GetClass())
      return false;

   auto draw_func = [&subpad, &object, &opt](Draw7Map_t::iterator &iter) -> bool {
      return iter->second.func(subpad, object, opt);
   };

   if (ScanProviderMap<Draw7Map_t, Draw7Map_t::iterator>(GetDraw7Map(), object->GetClass(), false, draw_func))
      return true;

   // TODO: need factory methods for that

   if (object->GetClass()->InheritsFrom("TLeaf"))
      gSystem->Load("libROOTTreeDrawProvider");
   else if (object->GetClass()->InheritsFrom(TObject::Class()))
      gSystem->Load("libROOTObjectDrawProvider");
   else if (object->GetClass()->InheritsFrom("ROOT::Experimental::RH1D") ||
            object->GetClass()->InheritsFrom("ROOT::Experimental::RH2D") ||
            object->GetClass()->InheritsFrom("ROOT::Experimental::RH3D"))
      gSystem->Load("libROOTHistDrawProvider");
   else
      return false;

   return ScanProviderMap<Draw7Map_t, Draw7Map_t::iterator>(GetDraw7Map(), object->GetClass(), true, draw_func);
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
