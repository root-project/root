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
namespace Browsable {

/** \class TObjectHolder
\ingroup rbrowser
\brief Holder of TObject instance. Should not be used very often, while ownership is undefined for it
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class TObjectHolder : public RHolder {
   TObject *fObj{nullptr};   ///<! plain holder without IO
   void *fAdjusted{nullptr}; ///<! pointer on real class returned by fObj->IsA()
   bool fOwner{false};       ///<! is TObject owner
protected:
   void *AccessObject() final { return fOwner ? nullptr : fObj; }
   void *TakeObject() final;
   RHolder *DoCopy() const final { return new TObjectHolder(fObj); }
   void ClearROOTOwnership(TObject *obj);
public:
   TObjectHolder(TObject *obj, bool owner = false)
   {
      fAdjusted = fObj = obj;
      fOwner = owner;
      if (fOwner && fObj)
         ClearROOTOwnership(fObj);
      if (fAdjusted) {
         auto offset = fObj->IsA()->GetBaseClassOffset(TObject::Class());
         if (offset > 0)
            fAdjusted = (char *) fAdjusted - offset;
      }
   }

   virtual ~TObjectHolder()
   {
      if (fOwner) delete fObj;
   }

   void Forget() final
   {
      fAdjusted = fObj = nullptr;
      fOwner = false;
   }

   const TClass *GetClass() const final { return fObj ? fObj->IsA() : nullptr; }
   const void *GetObject() const final { return fAdjusted; }
};


} // namespace Browsable
} // namespace ROOT


#endif
