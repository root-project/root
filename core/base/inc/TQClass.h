// @(#)root/base:$Id$
// Author: Valeriy Onuchin & Fons Rademakers   15/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQClass
#define ROOT_TQClass

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This is part of the ROOT implementation of the Qt object             //
// communication mechanism (see also                                    //
// http://www.troll.no/qt/metaobjects.html)                             //
//                                                                      //
// See TQObject for details.                                             //
//                                                                      //
// This implementation is provided by                                   //
// Valeriy Onuchin (onuchin@sirius.ihep.su).                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TQObject
#include "TQObject.h"
#endif
#ifndef ROOT_TClass
#include "TClass.h"
#endif

// This class makes it possible to have a single connection from
// all objects of the same class
class TQClass : public TQObject, public TClass {

private:
   TQClass(const TClass&) : TQObject(), TClass() {};
   TQClass& operator=(const TQClass&) { return *this; }
      
friend class TQObject;

public:
   TQClass(const char *name, Version_t cversion,
           const type_info &info, TVirtualIsAProxy *isa,
           ShowMembersFunc_t showmembers,
           const char *dfil = 0, const char *ifil = 0,
           Int_t dl = 0, Int_t il = 0) :
           TQObject(),
           TClass(name, cversion, info,isa,showmembers, dfil, ifil, dl, il) { }

   virtual ~TQClass() { Disconnect(); }

   ClassDef(TQClass,0)  // Class with connections
};


//---- Class Initialization Behavior --------------------------------------
//
// This Class and Function are automatically used for classes inheriting from
// TQObject. They make it possible to have a single connection from all
// objects of the same class.
namespace ROOT {
   class TDefaultInitBehavior;
   class TQObjectInitBehavior : public TDefaultInitBehavior {
   public:
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, TVirtualIsAProxy *isa,
                                  ShowMembersFunc_t show,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const
      {
         return new TQClass(cname, id, info, isa, show, dfil, ifil,dl, il);
      }
   };

   inline const TQObjectInitBehavior *DefineBehavior(TQObject*, TQObject*)
   {
      TQObjectInitBehavior *behave = new TQObjectInitBehavior;
      return behave;
   }
}

#endif

