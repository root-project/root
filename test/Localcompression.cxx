// @(#)root/test:$Id$
// Author: Rene Brun   19/08/96

#include "RVersion.h"
#include "TRandom.h"
#include "TDirectory.h"
#include "TProcessID.h"

#include "Localcompression.h"
#include <iostream>

ClassImp(TDummy)
ClassImp(TLarge)
ClassImp(TSmall)
ClassImp(TFloat)

////////////////////////////////////////////////////////////////////////////////
/// Create an TDummy.
TDummy::TDummy(Int_t size)
{
   fSize = size;
   fDummy = new Float_t[fSize];
   for(int i=0;i<fSize;++i) {
      Dummy_t dummy;
      for(int offset = 0; offset < 4; ++offset) {
         Float_t tmp = Float_t(gRandom->Rndm(1));
         if(tmp < 0.08167) {
            dummy.ch[offset] = 'a';
         } else if(tmp < 0.09659) {
            dummy.ch[offset] = 'b';
         } else if(tmp < 0.12441) {
            dummy.ch[offset] = 'c';
         } else if(tmp < 0.16694) {
            dummy.ch[offset] = 'd';
         } else if(tmp < 0.29396) {
            dummy.ch[offset] = 'e';
         } else if(tmp < 0.31624) {
            dummy.ch[offset] = 'f';
         } else if(tmp < 0.33639) {
            dummy.ch[offset] = 'g';
         } else if(tmp < 0.39733) {
            dummy.ch[offset] = 'h';
         } else if(tmp < 0.46699) {
            dummy.ch[offset] = 'i';
         } else if(tmp < 0.46852) {
            dummy.ch[offset] = 'j';
         } else if(tmp < 0.47624) {
            dummy.ch[offset] = 'k';
         } else if(tmp < 0.51649) {
            dummy.ch[offset] = 'l';
         } else if(tmp < 0.54055) {
            dummy.ch[offset] = 'm';
         } else if(tmp < 0.60804) {
            dummy.ch[offset] = 'n';
         } else if(tmp < 0.68311) {
            dummy.ch[offset] = 'o';
         } else if(tmp < 0.7024) {
            dummy.ch[offset] = 'p';
         } else if(tmp < 0.70335) {
            dummy.ch[offset] = 'q';
         } else if(tmp < 0.76322) {
            dummy.ch[offset] = 'r';
         } else if(tmp < 0.82649) {
            dummy.ch[offset] = 's';
         } else if(tmp < 0.91705) {
            dummy.ch[offset] = 't';
         } else if(tmp < 0.94463) {
            dummy.ch[offset] = 'u';
         } else if(tmp < 0.95441) {
            dummy.ch[offset] = 'v';
         } else if(tmp < 0.97801) {
            dummy.ch[offset] = 'w';
         } else if(tmp < 0.97951) {
            dummy.ch[offset] = 'x';
         } else if(tmp < 0.99925) {
            dummy.ch[offset] = 'y';
         } else {
            dummy.ch[offset] = 'z';
         }
//         std::cout << "dummy.ch[" << offset << "]=" << dummy.ch[offset] << ", ";//##
      }
      fDummy[i] = dummy.fp;
//      std::cout << "fDummy[" << i << "]=" << fDummy[i] << std::endl;//##
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create an TDummy.
TDummy::TDummy(const TDummy& dummy) : TObject(dummy)
{
   Float_t *intermediate = dummy.GetDummy();
   Int_t size = dummy.GetSize();
   fDummy = new Float_t[size];
   for(int i=0;i<size;++i)
      fDummy[i] = intermediate[i];
}

////////////////////////////////////////////////////////////////////////////////

TDummy::~TDummy()
{
   Clear();
   delete fDummy;
   fSize = 0;
}

////////////////////////////////////////////////////////////////////////////////

void TDummy::Clear(Option_t * /*option*/)
{
   TObject::Clear();
   for(int i=0;i<fSize;++i)
      fDummy[i] = 0;
}

/////////////////////////////////////////////////////////////////////////////////

void TDummy::Build()
{
   for(int i=0;i<fSize;++i) {
      Dummy_t dummy;
      for(int offset = 0; offset < 4; ++offset) {
         Float_t tmp = Float_t(gRandom->Rndm(1));
         if(tmp < 0.08167) {
            dummy.ch[offset] = 'a';
         } else if(tmp < 0.09659) {
            dummy.ch[offset] = 'b';
         } else if(tmp < 0.12441) {
            dummy.ch[offset] = 'c';
         } else if(tmp < 0.16694) {
            dummy.ch[offset] = 'd';
         } else if(tmp < 0.29396) {
            dummy.ch[offset] = 'e';
         } else if(tmp < 0.31624) {
            dummy.ch[offset] = 'f';
         } else if(tmp < 0.33639) {
            dummy.ch[offset] = 'g';
         } else if(tmp < 0.39733) {
            dummy.ch[offset] = 'h';
         } else if(tmp < 0.46699) {
            dummy.ch[offset] = 'i';
         } else if(tmp < 0.46852) {
            dummy.ch[offset] = 'j';
         } else if(tmp < 0.47624) {
            dummy.ch[offset] = 'k';
         } else if(tmp < 0.51649) {
            dummy.ch[offset] = 'l';
         } else if(tmp < 0.54055) {
            dummy.ch[offset] = 'm';
         } else if(tmp < 0.60804) {
            dummy.ch[offset] = 'n';
         } else if(tmp < 0.68311) {
            dummy.ch[offset] = 'o';
         } else if(tmp < 0.7024) {
            dummy.ch[offset] = 'p';
         } else if(tmp < 0.70335) {
            dummy.ch[offset] = 'q';
         } else if(tmp < 0.76322) {
            dummy.ch[offset] = 'r';
         } else if(tmp < 0.82649) {
            dummy.ch[offset] = 's';
         } else if(tmp < 0.91705) {
            dummy.ch[offset] = 't';
         } else if(tmp < 0.94463) {
            dummy.ch[offset] = 'u';
         } else if(tmp < 0.95441) {
            dummy.ch[offset] = 'v';
         } else if(tmp < 0.97801) {
            dummy.ch[offset] = 'w';
         } else if(tmp < 0.97951) {
            dummy.ch[offset] = 'x';
         } else if(tmp < 0.99925) {
            dummy.ch[offset] = 'y';
         } else {
            dummy.ch[offset] = 'z';
         }
//         std::cout << "dummy.ch[" << offset << "]=" << dummy.ch[offset] << ", ";//##
      }
      fDummy[i] = dummy.fp;
//      std::cout << "fDummy[" << i << "]=" << fDummy[i] << std::endl;//##
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create an TLarge.
TLarge::TLarge(Int_t size)
{
   fSize = size;
   fLarge = new Float_t[fSize];
   for(int i=0;i<fSize;++i) {
      if (i%6==0) fLarge[i] = gRandom->Rndm(1);
      else fLarge[i] = fLarge[i-1];
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create an TLarge.
TLarge::TLarge(const TLarge& large) : TObject(large)
{
   Float_t *intermediate = large.GetLarge();
   Int_t size = large.GetSize();
   fLarge = new Float_t[size];
   for(int i=0;i<size;++i)
      fLarge[i] = intermediate[i];
}

////////////////////////////////////////////////////////////////////////////////

TLarge::~TLarge()
{
   Clear();
   delete fLarge;
   fSize = 0;
}

////////////////////////////////////////////////////////////////////////////////

void TLarge::Clear(Option_t * /*option*/)
{
   TObject::Clear();
   for(int i=0;i<fSize;++i)
      fLarge[i] = 0;
}

/////////////////////////////////////////////////////////////////////////////////

void TLarge::Build()
{
   for(int i=0;i<fSize;++i) {
      if (i%6==0) fLarge[i] = gRandom->Rndm(1);
      else fLarge[i] = fLarge[i-1];
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Create an TSmall.
TSmall::TSmall(Int_t size)
{
   fSize  = size;
   fSmall = new Float_t[fSize];
   for(int i=0;i<fSize;++i) {
      if (i%6==0) fSmall[i] = gRandom->Rndm(1);
      fSmall[i] = fSmall[i-1];
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create an TSmall.
TSmall::TSmall(const TSmall& small) : TObject(small)
{
   Float_t *intermediate = small.GetSmall();
   Int_t size = small.GetSize();
   fSmall = new Float_t[size];
   for(int i=0;i<size;++i)
      fSmall[i] = intermediate[i];
}

////////////////////////////////////////////////////////////////////////////////

TSmall::~TSmall()
{
   Clear();
   delete fSmall;
   fSize = 0;
}

//////////////////////////////////////////////////////////////////////////////////

void TSmall::Build()
{
   for(int i=0;i<fSize;++i) {
      if (i%6==0) fSmall[i] = gRandom->Rndm(1);
      else fSmall[i] = fSmall[i-1];
   }
}

///////////////////////////////////////////////////////////////////////////////

void TSmall::Clear(Option_t * /*option*/)
{
   TObject::Clear();
   for(int i=0;i<fSize;++i)
      fSmall[i] = 0;
}

///////////////////////////////////////////////////////////////////////////////
/// Create an TFloat.
TFloat::TFloat(Int_t size)
{
   fSize  = size;
   fFloat = new Float_t[fSize];
   fFloat[0] = Float_t(gRandom->Rndm(1));
   for(int i=1;i<fSize;++i) {
      fFloat[i] = fFloat[0];
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create an TSmall.
TFloat::TFloat(const TFloat& afloat) : TObject(afloat)
{
   Float_t *intermediate = afloat.GetFloat();
   Int_t size = afloat.GetSize();
   fFloat = new Float_t[size];
   for(int i=0;i<size;++i)
      fFloat[i] = intermediate[i];
}

////////////////////////////////////////////////////////////////////////////////

TFloat::~TFloat()
{
   Clear();
   delete fFloat;
   fSize = 0;
}

//////////////////////////////////////////////////////////////////////////////////

void TFloat::Build()
{
   fFloat[0] = Float_t(gRandom->Rndm(1));
   for(int i=1;i<fSize;++i) {
      fFloat[i] = fFloat[0];
   }
}

///////////////////////////////////////////////////////////////////////////////

void TFloat::Clear(Option_t * /*option*/)
{
   TObject::Clear();
   for(int i=0;i<fSize;++i)
      fFloat[i] = 0;
}

