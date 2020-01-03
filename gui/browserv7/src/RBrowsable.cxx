/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RBrowsable.hxx"

#include "ROOT/RLogger.hxx"

#include "TClass.h"
#include "TBufferJSON.h"

#include <algorithm>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Browsable;
using namespace std::string_literals;


/////////////////////////////////////////////////////////////////////
/// Find item with specified name
/// Default implementation, should work for all

RElement::EContentKind RElement::GetContentKind(const std::string &kind)
{
   if (kind == "text") return kText;
   if ((kind == "image") || (kind == "image64")) return kImage;
   if (kind == "png") return kPng;
   if ((kind == "jpg") || (kind == "jpeg")) return kJpeg;
   if (kind == "filename") return kFileName;
   return kNone;
}

/////////////////////////////////////////////////////////////////////
/// Returns sub element

std::shared_ptr<RElement> RElement::GetSubElement(std::shared_ptr<RElement> &elem, const RElementPath_t &path)
{
   auto curr = elem;

   for (auto &itemname : path) {
      if (!curr)
         return nullptr;

      auto iter = curr->GetChildsIter();
      if (!iter || !iter->Find(itemname))
         return nullptr;

      curr = iter->GetElement();
   }

   return curr;
}



class RCompositeIter : public RLevelIter {
   int fIndx{-1};
   RComposite &fComp;

public:

   explicit RCompositeIter(RComposite &comp) : fComp(comp) {}
   virtual ~RCompositeIter() = default;

   /** Shift to next element */
   bool Next() override { fIndx++; return HasItem(); }

   /** Is there current element  */
   bool HasItem() const override { return (fIndx >= 0) &&  (fIndx < (int) fComp.GetChilds().size()); }

   /** Returns current element name  */
   std::string GetName() const override { return fComp.GetChilds()[fIndx]->GetName(); }

   /** If element may have childs: 0 - no, >0 - yes, -1 - maybe */
   int CanHaveChilds() const override { return fComp.GetChilds().size(); }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override { return fComp.GetChilds()[fIndx]; }

   /** Reset iterator to the first element, returns false if not supported */
   bool Reset() override { fIndx = -1; return true; }

};

/////////////////////////////////////////////////////////////////////
/// Create iterator for childs of composite

std::unique_ptr<RLevelIter> RComposite::GetChildsIter()
{
   return std::make_unique<RCompositeIter>(*this);
}


/////////////////////////////////////////////////////////////////////
/// Find item with specified name
/// Default implementation, should work for all

bool RLevelIter::Find(const std::string &name)
{
   if (!Reset()) return false;

   while (Next()) {
      if (GetName() == name)
         return true;
   }

   return false;
}



std::string RProvider::GetClassIcon(const std::string &classname)
{
   if (classname == "TTree" || classname == "TNtuple")
      return "sap-icon://tree"s;
   if (classname == "TDirectory" || classname == "TDirectoryFile")
      return "sap-icon://folder-blank"s;
   if (classname.find("TLeaf") == 0)
      return "sap-icon://e-care"s;

   return "sap-icon://electronic-medical-record"s;
}


RProvider::BrowseMap_t &RProvider::GetBrowseMap()
{
   static RProvider::BrowseMap_t sMap;
   return sMap;
}

RProvider::FileMap_t &RProvider::GetFileMap()
{
   static RProvider::FileMap_t sMap;
   return sMap;
}


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

std::shared_ptr<RElement> RProvider::Browse(std::unique_ptr<Browsable::RHolder> &object)
{
   auto &bmap = GetBrowseMap();

   auto cl = object->GetClass();

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
/// set top element for browsing

void RBrowsable::SetTopElement(std::shared_ptr<Browsable::RElement> elem)
{
   fTopElement = elem;

   SetWorkingDirectory("");
}

/////////////////////////////////////////////////////////////////////
/// set working directory relative to top element

void RBrowsable::SetWorkingDirectory(const std::string &strpath)
{
   auto path = DecomposePath(strpath);

   SetWorkingPath(path);
}

/////////////////////////////////////////////////////////////////////
/// set working directory relative to top element

void RBrowsable::SetWorkingPath(const RElementPath_t &path)
{
   fWorkingPath = path;
   fWorkElement = RElement::GetSubElement(fTopElement, path);

   ResetLastRequest();
}

/////////////////////////////////////////////////////////////////////
/// Reset all data correspondent to last request

void RBrowsable::ResetLastRequest()
{
   fLastAllChilds = false;
   fLastSortedItems.clear();
   fLastSortMethod.clear();
   fLastItems.clear();
   fLastPath.clear();
   fLastElement.reset();
}

/////////////////////////////////////////////////////////////////////////
/// Decompose path to elements
/// Returns array of names for each element in the path, first element either "/" or "."
/// If returned array empty - it is error

RElementPath_t RBrowsable::DecomposePath(const std::string &strpath)
{
   RElementPath_t arr;

   if (strpath.empty())
      return arr;

   std::string slash = "/";

   std::string::size_type previous = 0;
   if (strpath[0] == slash[0]) previous++;

   auto current = strpath.find(slash, previous);
   while (current != std::string::npos) {
      if (current > previous)
         arr.emplace_back(strpath.substr(previous, current - previous));
      previous = current + 1;
      current = strpath.find(slash, previous);
   }

   if (previous < strpath.length())
      arr.emplace_back(strpath.substr(previous));

   return arr;
}

/////////////////////////////////////////////////////////////////////////
/// Process browser request

bool RBrowsable::ProcessBrowserRequest(const RBrowserRequest &request, RBrowserReply &reply)
{
   if (gDebug > 0)
      printf("REQ: Do decompose path '%s'\n",request.path.c_str());

   auto path = DecomposePath(request.path);

   if ((path != fLastPath) || !fLastElement) {

      auto elem = RElement::GetSubElement(fWorkElement, path);
      if (!elem) return false;

      ResetLastRequest();

      fLastPath = path;
      fLastElement = elem;
   }

   // when request childs, always try to make elements
   if (fLastItems.empty()) {
      auto iter = fLastElement->GetChildsIter();
      if (!iter) return false;
      int id = 0;
      fLastAllChilds = true;

      while (iter->Next() && fLastAllChilds) {
         fLastItems.emplace_back(iter->CreateBrowserItem());
         if (id++ > 10000)
            fLastAllChilds = false;
      }

      fLastSortedItems.clear();
      fLastSortMethod.clear();
   }

   // create sorted array
   if ((fLastSortedItems.size() != fLastItems.size()) || (fLastSortMethod != request.sort)) {
      fLastSortedItems.resize(fLastItems.size(), nullptr);
      int id = 0;
      if (request.sort.empty()) {
         // no sorting, just move all folders up
         for (auto &item : fLastItems)
            if (item->IsFolder())
               fLastSortedItems[id++] = item.get();
         for (auto &item : fLastItems)
            if (!item->IsFolder())
               fLastSortedItems[id++] = item.get();
      } else {
         // copy items
         for (auto &item : fLastItems)
            fLastSortedItems[id++] = item.get();

         if (request.sort != "unsorted")
            std::sort(fLastSortedItems.begin(), fLastSortedItems.end(),
                      [request](const RBrowserItem *a, const RBrowserItem *b) { return a->Compare(b, request.sort); });
      }
      fLastSortMethod = request.sort;
   }

   int id = 0;
   for (auto &item : fLastSortedItems) {
      if (!request.filter.empty() && (item->GetName().compare(0, request.filter.length(), request.filter) != 0))
         continue;

      if ((id >= request.first) && ((request.number == 0) || (id < request.first + request.number)))
         reply.nodes.emplace_back(item);
      id++;
   }

   reply.first = request.first;
   reply.nchilds = id; // total number of childs

   return true;
}

/////////////////////////////////////////////////////////////////////////
/// Process browser request, returns string with JSON of RBrowserReply data

std::string RBrowsable::ProcessRequest(const RBrowserRequest &request)
{
   RBrowserReply reply;

   reply.path = request.path;
   reply.first = 0;
   reply.nchilds = 0;

   ProcessBrowserRequest(request, reply);

   return TBufferJSON::ToJSON(&reply, TBufferJSON::kSkipTypeInfo + TBufferJSON::kNoSpaces).Data();
}

/////////////////////////////////////////////////////////////////////////
/// Returns element with path, specified as string

std::shared_ptr<Browsable::RElement> RBrowsable::GetElement(const std::string &str)
{
   auto path = DecomposePath(str);

   return RElement::GetSubElement(fWorkElement, path);
}

/////////////////////////////////////////////////////////////////////////
/// Returns element with path, specified as RElementPath_t

std::shared_ptr<Browsable::RElement> RBrowsable::GetElementFromTop(const RElementPath_t &path)
{
   return RElement::GetSubElement(fTopElement, path);
}


