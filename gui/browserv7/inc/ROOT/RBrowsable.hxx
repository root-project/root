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

#include "TClass.h"

#include <memory>
#include <string>
#include <map>
#include <vector>

class TObject;

namespace ROOT {
namespace Experimental {

namespace Browsable {

class RLevelIter;


class RObject {
public:
   virtual ~RObject() = default;

   virtual const TClass *GetClass() const = 0;
   /** Returns pointer when external memory management is used */
   virtual const void *GetObject() const { return nullptr; }
   /** Returns pointer on existing shared_ptr<T> */
   virtual void *GetShared() const { return nullptr; }
   /** Returns pointer with ownership, normally unique_ptr */
   virtual void *TakeObject() { return nullptr; }
};

/** Holder of TObject without ownership */
class RTObjectHolder : public RObject {
   TObject* fObj{nullptr};
public:
   RTObjectHolder(TObject *obj) { fObj = obj; }
   virtual ~RTObjectHolder() = default;

   const TClass *GetClass() const final { return fObj->IsA(); }
   const void *GetObject() const final { return fObj; }
};


/** Holder of shared_ptr<T> */

template<class T>
class RShared : public RObject {
   std::shared_ptr<T> fShared;
public:
   RShared(T *obj) { fShared.reset(obj); }
   RShared(std::shared_ptr<T> obj) { fShared = obj; }
   RShared(std::shared_ptr<T> &&obj) { fShared = std::move(obj); }
   virtual ~RShared() = default;

   const TClass *GetClass() const final { return TClass::GetClass<T>(); }
   void *GetShared() const final { return &fShared; }
};

/** Holder of unique_ptr<T> */

template<class T>
class RUnique : public RObject {
   std::unique_ptr<T> fUnique;
public:
   RUnique(T *obj) { fUnique.reset(obj); }
   RUnique(std::unique_ptr<T> &&obj) { fUnique = std::move(obj); }
   virtual ~RUnique() = default;

   const TClass *GetClass() const final { return TClass::GetClass<T>(); }
   void *TakeObject() final { return fUnique.release(); }
};


template<class T>
std::shared_ptr<T> get_shared(std::unique_ptr<RObject> &ptr)
{
   if (!ptr->GetClass()->InheritsFrom(TClass::GetClass<T>())) return nullptr;
   auto pshared = ptr->GetShared();
   if (pshared)
      return *(static_cast<std::shared_ptr<T> *>(pshared));
   auto pobj = ptr->TakeObject();
   if (pobj) {
      std::shared_ptr<T> shared;
      shared.reset(static_cast<T *>(pobj));
      return shared;
   }

   return nullptr;
}

/** \class RElement
\ingroup rbrowser
\brief Basic element of RBrowsable hierarchy. Provides access to data, creates iterator if any
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RElement {
public:
   virtual ~RElement() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   virtual std::string GetName() const = 0;

   /** Title of RBrowsable (optional) */
   virtual std::string GetTitle() const { return ""; }

   /** Create iterator for childs elements if any */
   virtual std::unique_ptr<RLevelIter> GetChildsIter() { return nullptr; }

   virtual bool HasTextContent() const { return false; }

   virtual std::string GetTextContent() { return ""; }

   /** Access object */
   virtual std::unique_ptr<RObject> GetObject(bool /* plain */ = false) { return nullptr; }
};

/** \class RLevelIter
\ingroup rbrowser
\brief Iterator over single level hierarchy like TList
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RLevelIter {
public:
   virtual ~RLevelIter() = default;

   /** Shift to next element */
   virtual bool Next() { return false; }

   /** Reset iterator to the first element, returns false if not supported */
   virtual bool Reset() { return false; }

   /** Is there current element  */
   virtual bool HasItem() const { return false; }

   /** Returns current element name  */
   virtual std::string GetName() const { return ""; }

   virtual bool Find(const std::string &name);

   /** If element may have childs: 0 - no, >0 - yes, -1 - maybe */
   virtual int CanHaveChilds() const { return 0; }

   virtual std::unique_ptr<RBrowserItem> CreateBrowserItem()
   {
      return std::make_unique<RBrowserItem>(GetName(), CanHaveChilds());
   }

   /** Returns full information for current element */
   virtual std::shared_ptr<RElement> GetElement() { return nullptr; }
};


/** \class RProvider
\ingroup rbrowser
\brief Provider of different browsing methods for supported classes
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RProvider {

   using Map_t = std::map<const TClass*, std::shared_ptr<RProvider>>;
   using FileMap_t = std::multimap<std::string, std::shared_ptr<RProvider>>;

   static Map_t &GetBrowseMap();
   static FileMap_t &GetFileMap();

protected:

   virtual std::shared_ptr<RElement> DoOpenFile(const std::string & /*fullname*/) const { return nullptr; }

   virtual std::shared_ptr<RElement> DoBrowse(const TClass */*cl*/, const void */*object*/) const { return nullptr; }


public:
   virtual ~RProvider() = default;

   static void RegisterFile(const std::string &extension, std::shared_ptr<RProvider> provider);
   static void RegisterBrowse(const TClass *cl, std::shared_ptr<RProvider> provider);
   static void Unregister(std::shared_ptr<RProvider> provider);

   static std::shared_ptr<RElement> OpenFile(const std::string &extension, const std::string &fullname);
   static std::shared_ptr<RElement> Browse(const TClass *cl, const void *object);

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

   struct RLevel {
      std::string name;
      std::unique_ptr<Browsable::RLevelIter> iter;
      std::shared_ptr<Browsable::RElement> item;
      RLevel(const std::string &_name) : name(_name) {}
   };

   std::shared_ptr<Browsable::RElement> fItem; ///<! top-level item to browse
   std::vector<RLevel> fLevels;           ///<! navigated levels

   bool Navigate(const std::vector<std::string> &path);

   bool DecomposePath(const std::string &path, std::vector<std::string> &arr);

public:
   RBrowsable() = default;

   RBrowsable(std::shared_ptr<Browsable::RElement> item)
   {
      fItem = item;
   }

   virtual ~RBrowsable() = default;


   void SetTopItem(std::shared_ptr<Browsable::RElement> item)
   {
      fLevels.clear();
      fItem = item;
   }

   bool ProcessRequest(const RBrowserRequest &request, RBrowserReplyNew &reply);

   std::shared_ptr<Browsable::RElement> GetElement(const std::string &path);
};


} // namespace Experimental
} // namespace ROOT

#endif
