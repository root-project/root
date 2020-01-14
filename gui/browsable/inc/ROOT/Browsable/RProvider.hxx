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

namespace ROOT {
namespace Experimental {
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

protected:

   using FileFunc_t = std::function<std::shared_ptr<RElement>(const std::string &)>;
   using BrowseFunc_t = std::function<std::shared_ptr<RElement>(std::unique_ptr<RHolder> &)>;

   void RegisterFile(const std::string &extension, FileFunc_t func);
   void RegisterBrowse(const TClass *cl, BrowseFunc_t func);

private:

   struct StructBrowse { RProvider *provider{nullptr};  BrowseFunc_t func; };
   struct StructFile { RProvider *provider{nullptr};  FileFunc_t func; };

   using BrowseMap_t = std::map<const TClass*, StructBrowse>;
   using FileMap_t = std::multimap<std::string, StructFile>;

   static BrowseMap_t &GetBrowseMap();
   static FileMap_t &GetFileMap();
};


} // namespace Browsable
} // namespace Experimental
} // namespace ROOT

#endif
