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

using RElementPath_t = std::vector<std::string>;


namespace ROOT {
namespace Experimental {

namespace Browsable {

/** \class RHolder
\ingroup rbrowser
\brief Basic class for object holder of any kind. Could be used to transfer shared_ptr or unique_ptr or plain pointer
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RHolder {
protected:

   /** Returns pointer on existing shared_ptr<T> */
   virtual void *GetShared() const { return nullptr; }

   /** Returns pointer with ownership, normally via unique_ptr<T>::release() or tobj->Clone() */
   virtual void *TakeObject() { return nullptr; }

   /** Returns plain object pointer without care about ownership, should not be used often */
   virtual void *AccessObject() { return nullptr; }

   /** Create copy of container, works only when pointer can be shared */
   virtual RHolder *DoCopy() const { return nullptr; }

public:
   virtual ~RHolder() = default;

   /** Returns class of contained object */
   virtual const TClass *GetClass() const = 0;

   /** Returns direct (temporary) object pointer */
   virtual const void *GetObject() const = 0;

   template <class T>
   bool InheritsFrom() const
   {
      return TClass::GetClass<T>()->InheritsFrom(GetClass());
   }

   template <class T>
   bool CanCastTo() const
   {
      return const_cast<TClass *>(GetClass())->GetBaseClassOffset(TClass::GetClass<T>()) == 0;
   }

   /** Returns direct object pointer cast to provided class */

   template<class T>
   const T *Get() const
   {
      if (CanCastTo<T>())
         return (const T *) GetObject();

      return nullptr;
   }

   /** Clone container. Trivial for shared_ptr and TObject holder, does not work for unique_ptr */
   auto Copy() const { return std::unique_ptr<RHolder>(DoCopy()); }


   /** Returns unique_ptr of contained object */
   template<class T>
   std::unique_ptr<T> get_unique()
   {
      // ensure that direct inheritance is used
      if (!CanCastTo<T>())
         return nullptr;
      auto pobj = TakeObject();
      if (pobj) {
         std::unique_ptr<T> unique;
         unique.reset(static_cast<T *>(pobj));
         return unique;
      }
      return nullptr;
   }

   /** Returns shared_ptr of contained object */
   template<class T>
   std::shared_ptr<T> get_shared()
   {
      // ensure that direct inheritance is used
      if (!CanCastTo<T>())
         return nullptr;
      auto pshared = GetShared();
      if (pshared)
         return *(static_cast<std::shared_ptr<T> *>(pshared));

      // automatically convert unique pointer to shared
      return get_unique<T>();
   }

   /** Returns plains pointer on object without ownership, only can be used for TObjects */
   template<class T>
   T *get_object()
   {
      if (!CanCastTo<T>())
         return nullptr;

      return (T *) AccessObject();
   }
};


/** \class RShared<T>
\ingroup rbrowser
\brief Holder of with shared_ptr<T> instance. Should be used to transfer shared_ptr<T> in RBrowsable methods
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

template<class T>
class RShared : public RHolder {
   std::shared_ptr<T> fShared;   ///<! holder without IO
protected:
   void *GetShared() const final { return &fShared; }
   RHolder* DoCopy() const final { return new RShared<T>(fShared); }
public:
   RShared(T *obj) { fShared.reset(obj); }
   RShared(std::shared_ptr<T> obj) { fShared = obj; }
   RShared(std::shared_ptr<T> &&obj) { fShared = std::move(obj); }
   virtual ~RShared() = default;

   const TClass *GetClass() const final { return TClass::GetClass<T>(); }
   const void *GetObject() const final { return fShared.get(); }
};

/** \class RUnique<T>
\ingroup rbrowser
\brief Holder of with unique_ptr<T> instance. Should be used to transfer unique_ptr<T> in RBrowsable methods
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

template<class T>
class RUnique : public RHolder {
   std::unique_ptr<T> fUnique; ///<! holder without IO
protected:
   void *TakeObject() final { return fUnique.release(); }
public:
   RUnique(T *obj) { fUnique.reset(obj); }
   RUnique(std::unique_ptr<T> &&obj) { fUnique = std::move(obj); }
   virtual ~RUnique() = default;

   const TClass *GetClass() const final { return TClass::GetClass<T>(); }
   const void *GetObject() const final { return fUnique.get(); }
};


/** \class RAnyObjectHolder
\ingroup rbrowser
\brief Holder of any object instance. Normally used with TFile, where any object can be read. Normally RShread or RUnique should be used
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAnyObjectHolder : public RHolder {
   TClass *fClass{nullptr};   ///<! object class
   void *fObj{nullptr};       ///<! plain holder without IO
   bool fOwner{false};        ///<! is object owner
protected:
   void *AccessObject() final { return fOwner ? nullptr : fObj; }

   void *TakeObject() final
   {
      if (!fOwner)
         return nullptr;
      auto res = fObj;
      fObj = nullptr;
      fOwner = false;
      return res;
   }

   RHolder* DoCopy() const final
   {
      if (fOwner || !fObj || !fClass) return nullptr;
      return new RAnyObjectHolder(fClass, fObj, false);
   }

public:
   RAnyObjectHolder(TClass *cl, void *obj, bool owner = false) { fClass = cl; fObj = obj; fOwner = owner; }
   virtual ~RAnyObjectHolder()
   {
      if (fOwner)
         fClass->Destructor(fObj);
   }

   const TClass *GetClass() const final { return fClass; }
   const void *GetObject() const final { return fObj; }
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

   enum EContentKind {
      kNone,      ///< not recognized
      kText,      ///< "text" - plain text for code editor
      kImage,     ///< "image64" - base64 for supported image formats (png/gif/gpeg)
      kPng,       ///< "png" - plain png binary code, returned inside std::string
      kJpeg,      ///< "jpg" or "jpeg" - plain jpg binary code, returned inside std::string
      kFileName   ///< "filename" - file name if applicable
   };

   static EContentKind GetContentKind(const std::string &kind);

   virtual ~RElement() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   virtual std::string GetName() const = 0;

   /** Checks if element name match to provided value */
   virtual bool MatchName(const std::string &name) const { return name == GetName(); }

   /** Title of RBrowsable (optional) */
   virtual std::string GetTitle() const { return ""; }

   /** Create iterator for childs elements if any */
   virtual std::unique_ptr<RLevelIter> GetChildsIter() { return nullptr; }

   /** Returns element content, depends from kind. Can be "text" or "image64" */
   virtual std::string GetContent(const std::string & = "text") { return ""; }

   /** Access object */
   virtual std::unique_ptr<RHolder> GetObject() { return nullptr; }

   static std::shared_ptr<RElement> GetSubElement(std::shared_ptr<RElement> &elem, const RElementPath_t &path);
};

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
   virtual bool Next() = 0;

   /** Is there current element  */
   virtual bool HasItem() const = 0;

   /** Returns current element name  */
   virtual std::string GetName() const = 0;

   /** If element may have childs: 0 - no, >0 - yes, -1 - maybe */
   virtual int CanHaveChilds() const { return 0; }

   /** Returns full information for current element */
   virtual std::shared_ptr<RElement> GetElement() = 0;

   virtual std::unique_ptr<RBrowserItem> CreateBrowserItem()
   {
      return HasItem() ? std::make_unique<RBrowserItem>(GetName(), CanHaveChilds(), CanHaveChilds() > 0 ? "sap-icon://folder-blank" : "sap-icon://document") : nullptr;
   }

   /** Reset iterator to the first element, returns false if not supported */
   virtual bool Reset() { return false; }

   virtual bool Find(const std::string &name);

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
