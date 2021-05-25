/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_TObjectHolder
#define ROOT7_Browsable_TObjectHolder

#include <ROOT/Browsable/RHolder.hxx>

namespace ROOT {
namespace Experimental {
namespace Browsable {

/** \class TObjectHolder
\ingroup rbrowser
\brief Holder of TObject instance. Should not be used very often, while ownership is undefined for it
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class TObjectHolder : public RHolder {
   TObject* fObj{nullptr};   ///<! plain holder without IO
   bool fOwner;              ///<! is TObject owner
protected:
   void *AccessObject() final { return fOwner ? nullptr : fObj; }
   void *TakeObject() final;
   RHolder *DoCopy() const final { return new TObjectHolder(fObj); }
public:
   TObjectHolder(TObject *obj, bool owner = false) { fObj = obj; fOwner = owner; }
   virtual ~TObjectHolder()
   {
      if (fOwner) delete fObj;
   }

   const TClass *GetClass() const final { return fObj ? fObj->IsA() : nullptr; }
   const void *GetObject() const final { return fObj; }
};


} // namespace Browsable
} // namespace Experimental
} // namespace ROOT


#endif
