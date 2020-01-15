/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RProvider
#define ROOT7_Browsable_RProvider

#include <ROOT/Browsable/RElement.hxx>

#include <functional>
#include <map>
#include <memory>

class TVirtualPad;

namespace ROOT {
namespace Experimental {

class RPadBase;

namespace Browsable {

/** \class RProvider
\ingroup rbrowser
\brief Provider of different browsing methods for supported classes
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RProvider {

public:

   virtual ~RProvider();

   static std::string GetClassIcon(const std::string &classname);

   static std::shared_ptr<RElement> OpenFile(const std::string &extension, const std::string &fullname);
   static std::shared_ptr<RElement> Browse(std::unique_ptr<RHolder> &obj);
   static bool Draw6(TVirtualPad *subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt = "");
   static bool Draw7(std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt = "");

protected:

   using FileFunc_t = std::function<std::shared_ptr<RElement>(const std::string &)>;
   using BrowseFunc_t = std::function<std::shared_ptr<RElement>(std::unique_ptr<RHolder> &)>;
   using Draw6Func_t = std::function<bool(TVirtualPad *, std::unique_ptr<Browsable::RHolder> &, const std::string &)>;
   using Draw7Func_t = std::function<bool(std::shared_ptr<RPadBase> &, std::unique_ptr<Browsable::RHolder> &, const std::string &)>;

   void RegisterFile(const std::string &extension, FileFunc_t func);
   void RegisterBrowse(const TClass *cl, BrowseFunc_t func);
   void RegisterDraw6(const TClass *cl, Draw6Func_t func);
   void RegisterDraw7(const TClass *cl, Draw7Func_t func);

private:

   struct StructBrowse { RProvider *provider{nullptr};  BrowseFunc_t func; };
   struct StructFile { RProvider *provider{nullptr};  FileFunc_t func; };
   struct StructDraw6 { RProvider *provider{nullptr};  Draw6Func_t func; };
   struct StructDraw7 { RProvider *provider{nullptr};  Draw7Func_t func; };

   using BrowseMap_t = std::multimap<const TClass*, StructBrowse>;
   using FileMap_t = std::multimap<std::string, StructFile>;
   using Draw6Map_t = std::multimap<const TClass*, StructDraw6>;
   using Draw7Map_t = std::multimap<const TClass*, StructDraw7>;

   static BrowseMap_t &GetBrowseMap();
   static FileMap_t &GetFileMap();
   static Draw6Map_t &GetDraw6Map();
   static Draw7Map_t &GetDraw7Map();

   template<class Map_t>
   void CleanThis(Map_t &fmap)
   {
      auto fiter = fmap.begin();
      while (fiter != fmap.end()) {
         if (fiter->second.provider == this)
            fiter = fmap.erase(fiter);
         else
            fiter++;
      }
   }

};


} // namespace Browsable
} // namespace Experimental
} // namespace ROOT

#endif
