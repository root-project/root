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

#include <memory>
#include <string>
#include <map>
#include <vector>


class TClass;

namespace ROOT {
namespace Experimental {


class RBrowsableLevelIter;

/** \class RBrowsableElement
\ingroup rbrowser
\brief Basic information about RBrowsable
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RBrowsableElement {
public:
   virtual ~RBrowsableElement() = default;

   /** Class information, must be provided in derived classes */
   virtual const TClass *GetGlass() const = 0;

   /** Name of RBrowsable, must be provided in derived classes */
   virtual std::string GetName() const = 0;

   /** Title of RBrowsable (optional) */
   virtual std::string GetTitle() const { return ""; }

   /** Returns estimated number of childs (-1 not implemented, have to try create iterator */
   virtual int CanHaveChilds() const { return -1; }

   /** Create iterator for childs elements if any */
   virtual std::unique_ptr<RBrowsableLevelIter> GetChildsIter() { return nullptr; }

   virtual std::unique_ptr<RBrowserItem> CreateBrowserItem()
   {
      return std::make_unique<RBrowserItem>(GetName(), CanHaveChilds());
   }
};

/** \class RBrowsableLevelIter
\ingroup rbrowser
\brief Iterator over single level hierarchy like TList
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RBrowsableLevelIter {
public:
   virtual ~RBrowsableLevelIter() = default;

   /** Shift to next element */
   virtual bool Next() { return false; }

   /** Reset iterator to the first element, returns false if not supported */
   virtual bool Reset() { return false; }

   /** Is there current element  */
   virtual bool HasItem() const { return false; }

   /** Returns current element name  */
   virtual std::string GetName() const { return ""; }

   virtual bool Find(const std::string &name);

   /** Returns full information for current element */
   virtual std::unique_ptr<RBrowsableElement> GetElement() { return nullptr; }
};

/** \class RBrowsable
\ingroup rbrowser
\brief Way to browse (hopefully) everything in ROOT
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RBrowsable {

   struct RLevel {
      std::string name;
      std::unique_ptr<RBrowsableLevelIter> iter;
      std::unique_ptr<RBrowsableElement> item;
      RLevel(const std::string &_name) : name(_name) {}
   };

   std::unique_ptr<RBrowsableElement> fItem; ///<! top-level item to browse
   std::vector<RLevel> fLevels;           ///<! navigated levels

   bool Navigate(const std::vector<std::string> &path);

   bool DecomposePath(const std::string &path, std::vector<std::string> &arr);

public:
   RBrowsable() = default;

   RBrowsable(std::unique_ptr<RBrowsableElement> &&item)
   {
      fItem = std::move(item);
   }

   virtual ~RBrowsable() = default;


   void SetTopItem(std::unique_ptr<RBrowsableElement> &&item)
   {
      fLevels.clear();
      fItem = std::move(item);
   }

   bool ProcessRequest(const RBrowserRequest &request, RBrowserReplyNew &reply);
};


/** \class RBrowsableProvider
\ingroup rbrowser
\brief Provider of different browsing methods for supported classes
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RBrowsableProvider {

   using Map_t = std::map<const TClass*, std::shared_ptr<RBrowsableProvider>>;

   static Map_t &GetMap();

public:
   virtual ~RBrowsableProvider() = default;

   /** Returns supported class */
   virtual const TClass *GetSupportedClass() const = 0;

   /** Returns true if derived classes supported as well */
   virtual bool SupportDerivedClasses() const { return false; }


   static void Register(std::shared_ptr<RBrowsableProvider> provider);
   static std::shared_ptr<RBrowsableProvider> GetProvider(const TClass *cl, bool check_base = true);

};


} // namespace Experimental
} // namespace ROOT

#endif
