// Author: Victor Perevoztchikov (fine@bnl.gov)   01/03/2001
// Copyright(c) 2001 [BNL] Brookhaven National Laboratory. All right reserved
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "TDsKey.h"


//______________________________________________________________________________
TDsKey::TDsKey(const char *name,UInt_t *uk,int nk) : fUrr(nk)
{
  if (name) SetName(name);
  SetUrr(uk,nk);
}
//______________________________________________________________________________
TDsKey::TDsKey(const char *name,UInt_t uk) : fUrr(1)
{
  if (name) SetName(name);
  SetUrr(&uk,1);
}
//______________________________________________________________________________
TDsKey::TDsKey(UInt_t uRun,UInt_t uEvent) :fUrr(2)
{
  UInt_t u[2]; u[0]=uRun; u[1]=uEvent;
  int n = 1; if (u[1]) n=2;
  SetUrr(u,n);
}

//______________________________________________________________________________
void TDsKey::SetUrr(const UInt_t *uk,int nk)
{
  int n;
  fUrr[0] = 0;
  if (!uk) return;
  for (n=1;n<nk && uk[n]; n++){}
  fUrr.Set(n,(Int_t *)uk);
}
//______________________________________________________________________________
TDsKey &TDsKey::operator=( const TDsKey &from)
{
  SetName(from.GetName());
  fUrr = from.fUrr;
  return *this;
}
//______________________________________________________________________________
TDsKey &TDsKey::operator=( UInt_t from)
{
  SetUrr(&from,1);
  return *this;
}
//______________________________________________________________________________
TDsKey &TDsKey::operator=( const char *from)
{
  SetName(from);
  return *this;
}
//______________________________________________________________________________
Bool_t TDsKey::operator==(const TDsKey &from) const
{   
   // Compare two keys
   Bool_t res  =    ( fName == from.fName )
                 && ( fUrr.GetSize() == from.fUrr.GetSize() ) ;
                 
   Bool_t numMatch = kTRUE;
   int i = 0;
   for (; i < fUrr.GetSize(); i++) {
     if ( (*(TArrayI *)(&fUrr))[i] != (*(TArrayI *)(&from.fUrr))[i]) {
        numMatch =kFALSE;
        break;
     }
   }
   return ( res && numMatch ) ;
}
//______________________________________________________________________________
void  TDsKey::Update( const TDsKey &from, const char *name)
{
  fUrr = from.fUrr;
  if (name) SetName(name);
}
//______________________________________________________________________________
TString TDsKey::GetKey() const
{
  char ubuf[12];
  TString tk(fName);
  Int_t lUrr = fUrr.GetSize();
  for (int i=0;i<lUrr;i++){
   tk +=".";
   sprintf(ubuf,"%010u",(*(TArrayI *)(&fUrr))[i]);
   tk +=ubuf;
  }
  return tk;
}
//______________________________________________________________________________
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
//______________________________________________________________________________
UInt_t  TDsKey::GetSum() const
{
  UInt_t s = (*(TArrayI *)(&fUrr))[0];
  for (int i=1;i<fUrr.GetSize();i++) s^=(*(TArrayI *)(&fUrr))[i];
  return s;
}


