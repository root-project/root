// @(#)root/table:$Id$
// Author: Victor Perevoztchikov (perev@bnl.gov)   01/03/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2001 [BNL] Brookhaven National Laboratory.              *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDsKey
#define ROOT_TDsKey

#include "TString.h"
#include "TArrayI.h"

const UInt_t kUMAX = (UInt_t)(-1);

class TDsKey
{
private:
   TString fName;
   TArrayI fUrr;

public:
   TDsKey(const char *name=0,UInt_t *uk=0,int nk=1);
   TDsKey(const char *name,UInt_t uk);
   TDsKey(UInt_t uRun,UInt_t uEvent=0);
   virtual ~TDsKey(){}
   virtual  TDsKey &operator=( const TDsKey &from);
   virtual  TDsKey &operator=( UInt_t from);
   virtual  TDsKey &operator=( Int_t from){ *this=(UInt_t)from; return *this;}
   virtual  TDsKey &operator=( const char *from);
   virtual  Bool_t operator==(const TDsKey &from) const;
   virtual  UInt_t  operator[](Int_t i) const { return fUrr[i]; }
   virtual  void    Update(const TDsKey &from,const char *name=0);
   virtual  void    SetName(const char *name){fName=name;}
   virtual  const char *GetName() const {return fName;}
   virtual  TString GetKey() const;
   virtual  UInt_t  EventNumber() const { return (*this)[1];}
   virtual  UInt_t  RunNumber() const { return (*this)[0];}
   virtual  void    SetKey(const char *key);
   virtual  void    SetUrr(const UInt_t *key,int nk);
   virtual  UInt_t  GetSum() const;
   virtual  Int_t   EOK()    const { return (UInt_t)fUrr[0]==kUMAX;}
   virtual  Int_t   IsNull() const { return !fUrr[0];}

};


#endif
