// @(#)root/base:$Id$
// Author: Rene Brun   15/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TStringLong
ATTENTION: this class is obsolete. It's functionality has been taken
over by TString.

The long string class (unlimited number of chars in I/O).

This class redefines only the I/O member functions of TString.
It uses 4 bytes to store the string length (1 byte only for TString).
*/

#include "TStringLong.h"
#include "TBuffer.h"
#include "Bytes.h"

ClassImp(TStringLong);


////////////////////////////////////////////////////////////////////////////////
///constructor

TStringLong::TStringLong() : TString()
{
}

////////////////////////////////////////////////////////////////////////////////
///constructor

TStringLong::TStringLong(Ssiz_t ic) : TString(ic)
{
}

////////////////////////////////////////////////////////////////////////////////
///copy constructor

TStringLong::TStringLong(const TString& s) : TString(s)
{
}

////////////////////////////////////////////////////////////////////////////////
///copy constructor

TStringLong::TStringLong(const char* cs) : TString(cs)
{
}

////////////////////////////////////////////////////////////////////////////////
///constructor from a char*

TStringLong::TStringLong(const char* cs, Ssiz_t n) : TString(cs,n)
{
}

////////////////////////////////////////////////////////////////////////////////
///constructor from a char

TStringLong::TStringLong(char c) : TString(c)
{
}

////////////////////////////////////////////////////////////////////////////////
///constructor from a char

TStringLong::TStringLong(char c, Ssiz_t n) : TString(c,n)
{
}

////////////////////////////////////////////////////////////////////////////////
///constructor from a substring

TStringLong::TStringLong(const TSubString& substr) : TString(substr)
{
}

////////////////////////////////////////////////////////////////////////////////
///destructor

TStringLong::~TStringLong()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Fill buffer.

void TStringLong::FillBuffer(char *&buffer) const
{
   Int_t nchars = Length();
   tobuf(buffer, nchars);
   const char *data = GetPointer();
   for (Int_t i = 0; i < nchars; i++) buffer[i] = data[i];
   buffer += nchars;
}

////////////////////////////////////////////////////////////////////////////////
/// Read this string from the buffer.

void TStringLong::ReadBuffer(char *&buffer)
{
   UnLink();
   Zero();

   Int_t nchars;
   frombuf(buffer, &nchars);

   char *data = Init(nchars, nchars);

   for (Int_t i = 0; i < nchars; i++) frombuf(buffer, &data[i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the sizeof the string.

Int_t TStringLong::Sizeof() const
{
   return Length()+sizeof(Int_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a long (>255 characters) string object.

void TStringLong::Streamer(TBuffer &b)
{
   Int_t nwh;
   if (b.IsReading()) {
      b >> nwh;
      Clobber(nwh);
      char *data = GetPointer();
      data[nwh] = 0;
      SetSize(nwh);
      for (int i = 0; i < nwh; i++) b >> data[i];
   } else {
      nwh = Length();
      b << nwh;
      const char *data = GetPointer();
      for (int i = 0; i < nwh; i++) b << data[i];
   }
}
