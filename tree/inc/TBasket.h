// @(#)root/tree:$Name:  $:$Id: TBasket.h,v 1.6 2002/01/16 18:10:23 brun Exp $
// Author: Rene Brun   19/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBasket
#define ROOT_TBasket


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBasket                                                              //
//                                                                      //
// The TBasket objects are created at run time to collect TTree entries //
// in buffers. When a Basket is full, it is written to the file.        //
// The Basket is kept in memory if there is enough space.               //
//  (see the fMaxVirtualsize of TTree).                                 //
//                                                                      //
// The Basket class derives from TKey.                                  //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TKey
#include "TKey.h"
#endif


class TFile;
class TTree;
class TBranch;

class TBasket : public TKey {

protected:
    Int_t       fBufferSize;      //fBuffer length in bytes
    Int_t       fNevBufSize;      //Length in Int_t of fEntryOffset
    Int_t       fNevBuf;          //Number of entries in basket
    Int_t       fLast;            //Pointer to last used byte in basket
    Bool_t      fHeaderOnly;      //True when only the basket header must be read/written
    Int_t      *fDisplacement;    //![fNevBuf] Displacement of entries in fBuffer(TKey)
    Int_t      *fEntryOffset;     //[fNevBuf] Offset of entries in fBuffer(TKey)
    TBranch    *fBranch;          //Pointer to the basket support branch

public:
    TBasket();
    TBasket(const char *name, const char *title, TBranch *branch);
    virtual ~TBasket();

    virtual void    AdjustSize(Int_t newsize);
    virtual void    DeleteEntryOffset();
    virtual Int_t   DropBuffers();
    TBranch        *GetBranch() const {return fBranch;}
            Int_t   GetBufferSize() const {return fBufferSize;}
            Int_t  *GetDisplacement() const {return fDisplacement;}
            Int_t  *GetEntryOffset() const {return fEntryOffset;}
            Int_t   GetEntryPointer(Int_t Entry);
            Int_t   GetNevBuf() const {return fNevBuf;}
            Int_t   GetNevBufSize() const {return fNevBufSize;}
            Int_t   GetLast() const {return fLast;}
            Int_t   ReadBasketBuffers(Seek_t pos, Int_t len, TFile *file);
            Int_t   ReadBasketBytes(Seek_t pos, TFile *file);

            void    SetBranch(TBranch *branch) {fBranch = branch;}
            void    SetNevBufSize(Int_t n) {fNevBufSize=n;}
    virtual void    SetReadMode();
    virtual void    SetWriteMode();
    inline  void    Update(Int_t newlast) { Update(newlast,newlast); }; 
    virtual void    Update(Int_t newlast, Int_t skipped);
    virtual Int_t   WriteBuffer();

    ClassDef(TBasket,2)  //the TBranch buffers
};

#endif
