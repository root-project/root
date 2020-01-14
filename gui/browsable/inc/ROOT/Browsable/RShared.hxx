/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RShared
#define ROOT7_Browsable_RShared

#include <ROOT/Browsable/RHolder.hxx>

namespace ROOT {
namespace Experimental {
namespace Browsable {

/** \class RShared<T>
\ingroup rbrowser
\brief Holder of with shared_ptr<T> instance. Should be used to transfer shared_ptr<T> in browsable methods
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

} // namespace Browsable
} // namespace Experimental
} // namespace ROOT


#endif
