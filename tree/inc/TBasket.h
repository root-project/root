// @(#)root/tree:$Name:  $:$Id: TBasket.h,v 1.1.1.1 2000/05/16 17:00:45 rdm Exp $
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
    char        *fZipBuffer;      //Basket compressed buffer (if compression)
    Int_t       *fDisplacement;   //![fNevBuf] Displacement of entries in fBuffer(TKey)
    Int_t       *fEntryOffset;    //[fNevBuf] Offset of entries in fBuffer(TKey)
    TBranch     *fBranch;         //Pointer to the basket support branch

public:
    TBasket();
    TBasket(const char *name, const char *title, TBranch *branch);
    virtual ~TBasket();

    virtual void    AdjustSize(Int_t newsize);
    virtual Int_t   DropBuffers();
    TBranch        *GetBranch() {return fBranch;}
            Int_t   GetBufferSize() {return fBufferSize;}
            Int_t  *GetDisplacement() {return fDisplacement;}
            Int_t  *GetEntryOffset() {return fEntryOffset;}
    virtual Int_t   GetEntryPointer(Int_t Entry);
            char   *GetZipBuffer() {return fZipBuffer;}
            Int_t   GetNevBuf() {return fNevBuf;}
            Int_t   GetNevBufSize() {return fNevBufSize;}
            Int_t   GetLast() {return fLast;}
    virtual void    ReadBasketBuffers(Seek_t pos, Int_t len, TFile *file);
    virtual Int_t   ReadBasketBytes(Seek_t pos, TFile *file);

            void    SetNevBufSize(Int_t n) {fNevBufSize=n;}
    virtual void    SetReadMode();
    virtual void    SetWriteMode();
    inline  void    Update(Int_t newlast) { Update(newlast,newlast); }; 
    virtual void    Update(Int_t newlast, Int_t skipped);
    virtual Int_t   WriteBuffer();

    ClassDef(TBasket,2)  //the TBranch buffers
};

#endif
