// @(#)root/base:$Name$:$Id$
// Author: Rene Brun   15/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStringLong                                                          //
//                                                                      //
// ATTENTION: this class is obsolete. It's functionality has been taken //
// over by TString.                                                     //
//                                                                      //
// The long string class (unlimited number of chars in I/O).            //
//                                                                      //
// This class redefines only the I/O member functions of TString.       //
// It uses 4 bytes to store the string length (1 byte only for TString).//
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TStringLong.h"
#include "TBuffer.h"
#include "Bytes.h"

ClassImp(TStringLong)


//______________________________________________________________________________
TStringLong::TStringLong() : TString()
{
}

//______________________________________________________________________________
TStringLong::TStringLong(Ssiz_t ic) : TString(ic)
{
}

//______________________________________________________________________________
TStringLong::TStringLong(const TString& s) : TString(s)
{
}

//______________________________________________________________________________
TStringLong::TStringLong(const char* cs) : TString(cs)
{
}

//______________________________________________________________________________
TStringLong::TStringLong(const char* cs, Ssiz_t n) : TString(cs,n)
{
}

//______________________________________________________________________________
TStringLong::TStringLong(char c) : TString(c)
{
}

//______________________________________________________________________________
TStringLong::TStringLong(char c, Ssiz_t n) : TString(c,n)
{
}

//______________________________________________________________________________
TStringLong::TStringLong(const TSubString& substr) : TString(substr)
{
}

//______________________________________________________________________________
TStringLong::~TStringLong()
{
}

//______________________________________________________________________________
void TStringLong::FillBuffer(char *&buffer)
{
   Int_t nchars = Length();
   tobuf(buffer, nchars);
   for (Int_t i = 0; i < nchars; i++) buffer[i] = fData[i];
   buffer += nchars;
}

//______________________________________________________________________________
void TStringLong::ReadBuffer(char *&buffer)
{
   Pref()->UnLink();

   Int_t nchars;
   frombuf(buffer, &nchars);

   fData = TStringRef::GetRep(nchars, nchars)->Data();

   for (Int_t i = 0; i < nchars; i++) frombuf(buffer, &fData[i]);
}

//______________________________________________________________________________
Int_t TStringLong::Sizeof() const
{
   return Length()+sizeof(Int_t);
}

//_______________________________________________________________________
void TStringLong::Streamer(TBuffer &b)
{
   // Stream a long (>255 characters) string object.

   Int_t nwh;
   if (b.IsReading()) {
      b >> nwh;
      fData = TStringRef::GetRep(nwh, nwh)->Data();
      for (int i = 0; i < nwh; i++) b >> fData[i];
   } else {
      nwh = Length();
      b << nwh;
      for (int i = 0; i < nwh; i++) b << fData[i];
   }
}
