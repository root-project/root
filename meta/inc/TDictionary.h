// @(#)root/meta:$Name:  $:$Id: TDictionary.h,v 1.2 2000/08/25 21:45:22 rdm Exp $
// Author: Fons Rademakers   20/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TDictionary
#define ROOT_TDictionary

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDictionary                                                          //
//                                                                      //
// This class defines an abstract interface that must be implemented    //
// by all classes that contain dictionary information.                  //
//                                                                      //
// The dictionary is defined by the followling classes:                 //
// TDataType                              (typedef definitions)         //
// TGlobal                                (global variables)            //
// TFunction                              (global functions)            //
// TClass                                 (classes)                     //
//    TBaseClass                          (base classes)                //
//    TDataMember                         (class datamembers)           //
//    TMethod                             (class methods)               //
//       TMethodArg                       (method arguments)            //
//                                                                      //
// All the above classes implement the TDictionary abstract interface   //
// (note: the indentation shows aggregation not inheritance).           //
// The ROOT dictionary system provides a very extensive RTTI            //
// environment that facilitates a.o. object inspectors, object I/O,     //
// ROOT Trees, etc. Most of the type information is provided by the     //
// CINT C++ interpreter.                                                //
//                                                                      //
// TMethodCall                            (method call environment)     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#include "Property.h"

enum EProperty {
   kIsClass       = G__BIT_ISCLASS,
   kIsStruct      = G__BIT_ISSTRUCT,
   kIsUnion       = G__BIT_ISUNION,
   kIsEnum        = G__BIT_ISENUM,
   kIsTypedef     = G__BIT_ISTYPEDEF,
   kIsFundamental = G__BIT_ISFUNDAMENTAL,
   kIsAbstract    = G__BIT_ISABSTRACT,
   kIsVirtual     = G__BIT_ISVIRTUAL,
   kIsPureVirtual = G__BIT_ISPUREVIRTUAL,
   kIsPublic      = G__BIT_ISPUBLIC,
   kIsProtected   = G__BIT_ISPROTECTED,
   kIsPrivate     = G__BIT_ISPRIVATE,
   kIsPointer     = G__BIT_ISPOINTER,
   kIsArray       = G__BIT_ISARRAY,
   kIsStatic      = G__BIT_ISSTATIC,
   kIsDefault     = G__BIT_ISDEFAULT,
   kIsReference   = G__BIT_ISREFERENCE,
   kIsConstant    = G__BIT_ISCONSTANT
};


class TDictionary : public TObject {

public:
   TDictionary() { }
   virtual ~TDictionary() { }

   virtual const char *GetName() const = 0;
   virtual const char *GetTitle() const = 0;
   virtual Long_t      Property() const = 0;

   virtual Int_t       Compare(const TObject *obj) const = 0;
   virtual ULong_t     Hash() const = 0;
   virtual Bool_t      IsSortable() const { return kTRUE; }

   ClassDef(TDictionary,0)  //ABC defining interface to dictionary
};

#endif
