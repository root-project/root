/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RAnyObjectHolder
#define ROOT7_Browsable_RAnyObjectHolder

#include <ROOT/Browsable/RHolder.hxx>

namespace ROOT {
namespace Experimental {
namespace Browsable {

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


} // namespace Browsable
} // namespace Experimental
} // namespace ROOT


#endif
