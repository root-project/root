// @(#)root/tree:$Name$:$Id$
// Author: Rene Brun   27/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafObject
#define ROOT_TLeafObject


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafObject                                                          //
//                                                                      //
// A TLeaf for a general object derived from TObject.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TLeaf
#include "TLeaf.h"
#endif

class TClass;
class TMethodCall;

class TLeafObject : public TLeaf {

protected:
    TClass      *fClass;          //pointer to class
    void        **fObjAddress;    //Address of Pointer to object

public:
    TLeafObject();
    TLeafObject(const char *name, const char *type);
    virtual ~TLeafObject();

    virtual void    FillBasket(TBuffer &b);
    TClass          *GetClass() {return fClass;}
    TMethodCall     *GetMethodCall(char *name);
    TObject         *GetObject() {return (TObject*)(*fObjAddress);}
    const char      *GetTypeName() const ;
    virtual void    ReadBasket(TBuffer &b);
    virtual void    SetAddress(void *add=0);

    ClassDef(TLeafObject,0)  //A TLeaf for a general object derived from TObject.
};

#endif
