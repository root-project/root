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
#include <functional>

class TObject;

namespace ROOT {
namespace Experimental {

namespace Browsable {

/** \class RObject
\ingroup rbrowser
\brief Basic class for object holder of any kind. Could be used to transfer shared_ptr or unique_ptr or plain pointer
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RObject {
protected:

   /** Returns pointer when external memory management is used */
   virtual const void *GetObject() const { return nullptr; }
   /** Returns pointer on existing shared_ptr<T> */
   virtual void *GetShared() const { return nullptr; }
   /** Returns pointer with ownership, normally via unique_ptr<T>::release() */
   virtual void *TakeObject() { return nullptr; }

public:
   virtual ~RObject() = default;

   virtual const TClass *GetClass() const = 0;

   template<class T>
   std::shared_ptr<T> get_shared()
   {
      if (!GetClass()->InheritsFrom(TClass::GetClass<T>()))
         return nullptr;
      auto pshared = GetShared();
      if (pshared)
         return *(static_cast<std::shared_ptr<T> *>(pshared));
      auto pobj = TakeObject();
      if (pobj) {
         std::shared_ptr<T> shared;
         shared.reset(static_cast<T *>(pobj));
         return shared;
      }

      return nullptr;
   }

   template<class T>
   T *get_object()
   {
      if (!GetClass()->InheritsFrom(TClass::GetClass<T>()))
         return nullptr;

      return (T *) GetObject();
   }
};


/** \class RTObjectHolder
\ingroup rbrowser
\brief Holder of TObject instance. Should not be used very often, while ownership is undefined for it
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RTObjectHolder : public RObject {
   TObject* fObj{nullptr};   ///<! plain holder without IO
protected:
   const void *GetObject() const final { return fObj; }
public:
   RTObjectHolder(TObject *obj) { fObj = obj; }
   virtual ~RTObjectHolder() = default;

   const TClass *GetClass() const final { return fObj->IsA(); }
};

/** \class RAnyObjectHolder
\ingroup rbrowser
\brief Holder of TObject instance. Should not be used very often, while ownership is undefined for it
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAnyObjectHolder : public RObject {
   const TClass *fClass{nullptr};  ///<! object class
   void* fObj{nullptr};            ///<! plain holder without IO
   bool fOwner{false};             ///<! is object owner
protected:
   void *TakeObject() final
   {
      void *res = fObj;
      if (fOwner)
         fObj = nullptr;
      else
         res = nullptr;
      return res;
   }
   const void *GetObject() const final { return fOwner ? nullptr : fObj; }

public:
   RAnyObjectHolder(const TClass *cl, void *obj, bool owner = false) { fClass = cl; fObj = obj; fOwner = owner; }
   virtual ~RAnyObjectHolder()
   {
      if (fOwner)
         const_cast<TClass *>(fClass)->Destructor(fObj);
   }

   const TClass *GetClass() const final { return fClass; }
};



/** \class RShared<T>
\ingroup rbrowser
\brief Holder of with shared_ptr<T> instance. Should be used to transfer shared_ptr<T> in RBrowsable methods
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

template<class T>
class RShared : public RObject {
   std::shared_ptr<T> fShared;   ///<! holder without IO
protected:
   void *GetShared() const final { return &fShared; }
public:
   RShared(T *obj) { fShared.reset(obj); }
   RShared(std::shared_ptr<T> obj) { fShared = obj; }
   RShared(std::shared_ptr<T> &&obj) { fShared = std::move(obj); }
   virtual ~RShared() = default;

   const TClass *GetClass() const final { return TClass::GetClass<T>(); }
};

/** \class RUnique<T>
\ingroup rbrowser
\brief Holder of with unique_ptr<T> instance. Should be used to transfer unique_ptr<T> in RBrowsable methods
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

template<class T>
class RUnique : public RObject {
   std::unique_ptr<T> fUnique; ///<! holder without IO
protected:
   void *TakeObject() final { return fUnique.release(); }
public:
   RUnique(T *obj) { fUnique.reset(obj); }
   RUnique(std::unique_ptr<T> &&obj) { fUnique = std::move(obj); }
   virtual ~RUnique() = default;

   const TClass *GetClass() const final { return TClass::GetClass<T>(); }
};


class RLevelIter;


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
\brief Iterator over single level hierarchy like any array, keys list, ...
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

public:

   virtual ~RProvider();

   static std::shared_ptr<RElement> OpenFile(const std::string &extension, const std::string &fullname);
   static std::shared_ptr<RElement> Browse(const TClass *cl, const void *object);

protected:

   using FileFunc_t = std::function<std::shared_ptr<RElement>(const std::string &)>;
   using BrowseFunc_t = std::function<std::shared_ptr<RElement>(const TClass *cl, const void *object)>;

   void RegisterFile(const std::string &extension, FileFunc_t provider);
   void RegisterBrowse(const TClass *cl, BrowseFunc_t provider);

private:

   using BrowseMap_t = std::map<const TClass*, BrowseFunc_t>;
   using FileMap_t = std::multimap<std::string, FileFunc_t>;


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
