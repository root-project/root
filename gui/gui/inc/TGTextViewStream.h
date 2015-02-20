// @(#)root/gui:$Id$

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextViewStream
#define ROOT_TGTextViewStream

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextViewStream                                                     //
//                                                                      //
// A TGTextViewStream is a text viewer widget. It is a specialization   //
// of TGTextView and std::ostream, and it uses a TGTextViewStreamBuf,   //
// who inherits from std::streambuf, allowing to stream text directly   //
// to the text view in a cout-like fashion                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGTextView
#include "TGTextView.h"
#endif
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
   virtual int overflow(int = traits::eof());

public:
   TGTextViewStreamBuf(TGTextView *textview);
   virtual ~TGTextViewStreamBuf() { }

   ClassDef(TGTextViewStreamBuf, 0) // Specialization of std::streambuf
};


class TGTextViewostream : public TGTextView, public std::ostream
{
protected:
   TGTextViewStreamBuf fStreambuffer;

public:
   TGTextViewostream(const TGWindow* parent = 0, UInt_t w = 1, UInt_t h = 1,
                     Int_t id = -1, UInt_t sboptions = 0,
                     Pixel_t back = TGTextView::GetWhitePixel());
   TGTextViewostream(const TGWindow *parent, UInt_t w, UInt_t h,
                     TGText *text, Int_t id, UInt_t sboptions, ULong_t back);
   TGTextViewostream(const TGWindow *parent, UInt_t w, UInt_t h,
                     const char *string, Int_t id, UInt_t sboptions,
                     ULong_t back);
   virtual ~TGTextViewostream() { }

   ClassDef(TGTextViewostream, 0) // Specialization of TGTextView and std::ostream
};

#endif
