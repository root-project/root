/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RHolder
#define ROOT7_Browsable_RHolder

#include "TClass.h"

#include <memory>

namespace ROOT {

class RLogChannel;

/// Log channel for Browsable diagnostics.
ROOT::RLogChannel &BrowsableLog(); // implemented in RElement.cxx

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

   /** Clear all pointers without performing cleanup */
   virtual void Forget() {}

   template <class T>
   bool InheritsFrom() const
   {
      return TClass::GetClass<T>()->InheritsFrom(GetClass());
   }

   template <class T>
   bool CanCastTo() const
   {
      return const_cast<TClass *>(GetClass())->GetBaseClassOffset(TClass::GetClass<T>()) >= 0;
   }

   /** Returns direct object pointer cast to provided class */

   template<class T>
   const T *Get() const
   {
      auto offset = const_cast<TClass *>(GetClass())->GetBaseClassOffset(TClass::GetClass<T>());
      if (offset >= 0)
         return (const T *) ((char *) GetObject() + offset);

      return nullptr;
   }

   /** Clone container. Trivial for shared_ptr and TObject holder, does not work for unique_ptr */
   auto Copy() const { return std::unique_ptr<RHolder>(DoCopy()); }

   /** Returns unique_ptr of contained object */
   template<class T>
   std::unique_ptr<T> get_unique()
   {
      // ensure that direct inheritance is used
      auto offset = const_cast<TClass *>(GetClass())->GetBaseClassOffset(TClass::GetClass<T>());
      if (offset < 0)
         return nullptr;
      auto pobj = TakeObject();
      if (pobj) {
         std::unique_ptr<T> unique;
         unique.reset((T *)((char *) pobj + offset));
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
      auto offset = const_cast<TClass *>(GetClass())->GetBaseClassOffset(TClass::GetClass<T>());
      if (offset < 0)
         return nullptr;

      return (T *) ((char *)AccessObject() + offset);
   }
};

} // namespace Browsable
} // namespace ROOT


#endif
