/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowsable
#define ROOT7_RBrowsable


#include <ROOT/RBrowserItem.hxx>

#include <ROOT/Browsable/RHolder.hxx>
#include <ROOT/Browsable/RElement.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>

#include "TClass.h"

#include <memory>
#include <string>
#include <map>
#include <vector>
#include <functional>

class TObject;


namespace ROOT {
namespace Experimental {

namespace Browsable {




/** \class RComposite
\ingroup rbrowser
\brief Composite elements - combines Basic element of RBrowsable hierarchy. Provides access to data, creates iterator if any
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-11-22
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RComposite : public RElement {

   std::string fName;
   std::string fTitle;
   std::vector<std::shared_ptr<RElement>> fChilds;

public:

   RComposite(const std::string &name, const std::string &title = "") : RElement(), fName(name), fTitle(title) {}

   virtual ~RComposite() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override { return fName; }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override { return fTitle; }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override;

   void Add(std::shared_ptr<RElement> elem) { fChilds.emplace_back(elem); }

   auto &GetChilds() const { return fChilds; }
};



/** \class RWrapper
\ingroup rbrowser
\brief Wrapper for other element - to provide different name
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-11-22
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RWrapper : public RElement {
   std::string fName;
   std::shared_ptr<RElement> fElem;

public:
   RWrapper() = default;

   RWrapper(const std::string &name, std::shared_ptr<RElement> elem) : fName(name), fElem(elem) {}

   virtual ~RWrapper() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override { return fName; }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override { return fElem->GetTitle(); }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override { return fElem->GetChildsIter(); }

   /** Returns element content, depends from kind. Can be "text" or "image64" */
   std::string GetContent(const std::string &kind = "text") override { return fElem->GetContent(kind); }

   /** Access object */
   std::unique_ptr<RHolder> GetObject() override { return fElem->GetObject(); }
};


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
   static std::shared_ptr<RElement> Browse(std::unique_ptr<Browsable::RHolder> &obj);

protected:

   using FileFunc_t = std::function<std::shared_ptr<RElement>(const std::string &)>;
   using BrowseFunc_t = std::function<std::shared_ptr<RElement>(std::unique_ptr<Browsable::RHolder> &)>;

   void RegisterFile(const std::string &extension, FileFunc_t func);
   void RegisterBrowse(const TClass *cl, BrowseFunc_t func);

private:

   struct StructBrowse { RProvider *provider;  BrowseFunc_t func; };
   struct StructFile { RProvider *provider;  FileFunc_t func; };

   using BrowseMap_t = std::map<const TClass*, StructBrowse>;
   using FileMap_t = std::multimap<std::string, StructFile>;

   static BrowseMap_t &GetBrowseMap();
   static FileMap_t &GetFileMap();

};

} // namespace Browsable


/** \class RBrowsable
\ingroup rbrowser
\brief Way to browse (hopefully) everything in ROOT
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RBrowsable {

   std::shared_ptr<Browsable::RElement> fTopElement;    ///<! top element for the RBrowsable

   RElementPath_t  fWorkingPath;                        ///<! path showed in Breadcrumb
   std::shared_ptr<Browsable::RElement> fWorkElement;   ///<! main element used for working in browser dialog

   RElementPath_t fLastPath;                             ///<! path to last used element
   std::shared_ptr<Browsable::RElement> fLastElement;    ///<! last element used in request
   std::vector<std::unique_ptr<RBrowserItem>> fLastItems; ///<! created browser items - used in requests
   bool fLastAllChilds{false};                           ///<! if all chlds were extracted
   std::vector<const RBrowserItem *> fLastSortedItems;   ///<! sorted child items, used in requests
   std::string fLastSortMethod;                          ///<! last sort method

   RElementPath_t DecomposePath(const std::string &path);

   void ResetLastRequest();

   bool ProcessBrowserRequest(const RBrowserRequest &request, RBrowserReply &reply);

public:
   RBrowsable() = default;

   RBrowsable(std::shared_ptr<Browsable::RElement> elem) { SetTopElement(elem); }

   virtual ~RBrowsable() = default;

   void SetTopElement(std::shared_ptr<Browsable::RElement> elem);

   void SetWorkingDirectory(const std::string &strpath);
   void SetWorkingPath(const RElementPath_t &path);

   const RElementPath_t &GetWorkingPath() const { return fWorkingPath; }

   std::string ProcessRequest(const RBrowserRequest &request);

   std::shared_ptr<Browsable::RElement> GetElement(const std::string &str);
   std::shared_ptr<Browsable::RElement> GetElementFromTop(const RElementPath_t &path);
};


} // namespace Experimental
} // namespace ROOT

#endif
