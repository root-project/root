// @(#)root/base:$Name:  $:$Id: TObjNum.h,v 1.1 2002/12/04 12:13:32 rdm Exp $
// Author: Fons Rademakers   02/12/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObjNum
#define ROOT_TObjNum


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjNum<T>                                                           //
//                                                                      //
// Wrap basic data type in a TObject.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif

#if defined(R__ANSISTREAM)
#   include <iostream>
    using std::cout;
    using std::endl;
#else
#   include <iostream.h>
#endif


template <class T> class TObjNum : public TObject {

private:
   T     fNum;      // number wrapped in TObject

public:
   TObjNum() { fNum = 0; }
   TObjNum(T num) : fNum(num) { }

   void    SetNum(T num) { fNum = num; }
   T       GetNum() const { return fNum; }
   void    Print(Option_t *) const { cout << fNum << endl; }
   ULong_t Hash() const { return TMath::Hash(&fNum, sizeof(T)); }
   Bool_t  IsEqual(const TObject *obj) const
      { return fNum == dynamic_cast<const TObjNum<T>*>(obj)->fNum; }
   Bool_t  IsSortable() const { return kTRUE; }
   Int_t   Compare(const TObject *obj) const
      { return (fNum > dynamic_cast<const TObjNum<T>*>(obj)->fNum) ? 1 :
               ((fNum < dynamic_cast<const TObjNum<T>*>(obj)->fNum) ? -1 : 0); }

   ClassDef(TObjNum,1)  //Basic type wrapped in a TObject
};

typedef TObjNum<Char_t>      TObjNumC;
typedef TObjNum<UChar_t>     TObjNumUC;
typedef TObjNum<Short_t>     TObjNumS;
typedef TObjNum<UShort_t>    TObjNumUS;
typedef TObjNum<Int_t>       TObjNumI;
typedef TObjNum<UInt_t>      TObjNumUI;
typedef TObjNum<Long_t>      TObjNumL;
typedef TObjNum<ULong_t>     TObjNumUL;
typedef TObjNum<Float_t>     TObjNumF;
typedef TObjNum<Double_t>    TObjNumD;
typedef TObjNum<void*>       TObjPtr;

#endif
