// @(#)root/tree:$Name:  $:$Id: TLeafElement.h,v 1.4 2000/12/13 15:13:55 brun Exp $
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafElement
#define ROOT_TLeafElement


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafElement                                                          //
//                                                                      //
// A TLeaf for a general object derived from TObject.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TLeaf
#include "TLeaf.h"
#endif

class TClass;
class TMethodCall;
class TStreamerInfo;
class TStreamerElement;

class TLeafElement : public TLeaf {

protected:
    TClass             *fClass;        //! pointer to class
    void               *fObjAddress;   //! Address of Pointer to object
    Bool_t              fVirtual;      //! Support for Virtuality
    Int_t               fID;           //element serial number in fInfo
    TStreamerInfo      *fInfo;         //!Pointer to StreamerInfo
    TStreamerElement   *fElement;      //!Pointer to StreamerElement
    
public:
    TLeafElement();
    TLeafElement(TStreamerInfo *sinfo, TStreamerElement *element, Int_t id, const char *type);
    virtual ~TLeafElement();

    virtual void    FillBasket(TBuffer &b);
    TClass         *GetClass() const {return fClass;}
    TMethodCall    *GetMethodCall(const char *name);
    const char     *GetTypeName() const ;
    Bool_t          IsVirtual() const {return fVirtual;}
    virtual void    ReadBasket(TBuffer &b);
    virtual void    SetAddress(void *add=0);
    virtual void    SetVirtual(Bool_t virt=kTRUE) {fVirtual=virt;}
    
    ClassDef(TLeafElement,2)  //A TLeaf for a general object derived from TObject.
};

#endif
