// @(#)root/table:$Id$
// Author: Victor Perevoztchikov (fine@bnl.gov)   01/03/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2001 [BNL] Brookhaven National Laboratory.              *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "TDsKey.h"


////////////////////////////////////////////////////////////////////////////////
///to be documented

TDsKey::TDsKey(const char *name,UInt_t *uk,int nk) : fUrr(nk)
{
   if (name) SetName(name);
   SetUrr(uk,nk);
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

TDsKey::TDsKey(const char *name,UInt_t uk) : fUrr(1)
{
   if (name) SetName(name);
   SetUrr(&uk,1);
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

TDsKey::TDsKey(UInt_t uRun,UInt_t uEvent) :fUrr(2)
{
   UInt_t u[2]; u[0]=uRun; u[1]=uEvent;
   int n = 1; if (u[1]) n=2;
   SetUrr(u,n);
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

void TDsKey::SetUrr(const UInt_t *uk,int nk)
{
   int n;
   fUrr[0] = 0;
   if (!uk) return;
   for (n=1;n<nk && uk[n]; n++){}
   fUrr.Set(n,(Int_t *)uk);
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

TDsKey &TDsKey::operator=( const TDsKey &from)
{
   SetName(from.GetName());
   fUrr = from.fUrr;
   return *this;
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

TDsKey &TDsKey::operator=( UInt_t from)
{
   SetUrr(&from,1);
   return *this;
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

TDsKey &TDsKey::operator=( const char *from)
{
   SetName(from);
   return *this;
}
////////////////////////////////////////////////////////////////////////////////
/// Compare two keys

Bool_t TDsKey::operator==(const TDsKey &from) const
{
   Bool_t res  =    ( fName == from.fName )
                 && ( fUrr.GetSize() == from.fUrr.GetSize() ) ;

   Bool_t numMatch = kTRUE;
   int i = 0;
   for (; i < fUrr.GetSize(); i++) {
      if ( fUrr[i] != from.fUrr[i] )  {
         numMatch =kFALSE;
         break;
      }
   }
   return ( res && numMatch ) ;
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

void  TDsKey::Update( const TDsKey &from, const char *name)
{
   fUrr = from.fUrr;
   if (name) SetName(name);
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

TString TDsKey::GetKey() const
{
   char ubuf[12];
   TString tk(fName);
   Int_t lUrr = fUrr.GetSize();
   for (int i=0;i<lUrr;i++){
      tk +=".";
      snprintf(ubuf,12,"%010u",fUrr[i]);
      tk +=ubuf;
   }
   return tk;
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

void TDsKey::SetKey(const char *key)
{
   const char *cc;
   int n = strchr(key,'.') - key;
   assert(n>0 && n<100);
   fName.Replace(0,999,key,n);
   Int_t i = 0;
   for (cc=key+n;*cc=='.'; cc+=11,i++)
      fUrr.AddAt(strtoul(cc+1,0,10),i);
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

UInt_t  TDsKey::GetSum() const
{
   UInt_t s = fUrr[0];
   for (int i=1;i<fUrr.GetSize();i++) s^=fUrr[i];
   return s;
}


