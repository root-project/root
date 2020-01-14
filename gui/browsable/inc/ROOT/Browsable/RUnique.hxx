/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RUnique
#define ROOT7_Browsable_RUnique

#include <ROOT/Browsable/RHolder.hxx>

namespace ROOT {
namespace Experimental {
namespace Browsable {

/** \class RUnique<T>
\ingroup rbrowser
\brief Holder of with unique_ptr<T> instance. Should be used to transfer unique_ptr<T> in browsable methods
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

} // namespace Browsable
} // namespace Experimental
} // namespace ROOT


#endif
