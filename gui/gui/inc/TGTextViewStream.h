// @(#)root/gui:$Id$

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextViewStream
#define ROOT_TGTextViewStream


#include "TGTextView.h"
#include <vector>
#include <streambuf>
#include <iostream>

#if defined (R__WIN32) && defined (__MAKECINT__)
typedef basic_streambuf<char, char_traits<char> > streambuf;
#endif

class TGTextViewStreamBuf : public std::streambuf
{
private:
   TGTextView *fTextView;
   std::vector<char> fLinebuffer;

protected:
   std::vector<char> fInputbuffer;
   typedef std::char_traits<char> traits;
   int overflow(int = traits::eof()) override;

public:
   TGTextViewStreamBuf(TGTextView *textview);
   virtual ~TGTextViewStreamBuf() {}

   ClassDef(TGTextViewStreamBuf, 0) // Specialization of std::streambuf
};


class TGTextViewostream : public TGTextView, public std::ostream
{
protected:
   TGTextViewStreamBuf fStreambuffer;

public:
   TGTextViewostream(const TGWindow* parent = nullptr, UInt_t w = 1, UInt_t h = 1,
                     Int_t id = -1, UInt_t sboptions = 0,
                     Pixel_t back = TGTextView::GetWhitePixel());
   TGTextViewostream(const TGWindow *parent, UInt_t w, UInt_t h,
                     TGText *text, Int_t id, UInt_t sboptions, ULong_t back);
   TGTextViewostream(const TGWindow *parent, UInt_t w, UInt_t h,
                     const char *string, Int_t id, UInt_t sboptions,
                     ULong_t back);
   virtual ~TGTextViewostream() {}

   ClassDefOverride(TGTextViewostream, 0) // Specialization of TGTextView and std::ostream
};

#endif
