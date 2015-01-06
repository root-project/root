// @(#)root/gui:$Id$

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TGTextViewStream.h"
#include "TSystem.h"

ClassImp(TGTextViewStreamBuf)

//______________________________________________________________________________
TGTextViewStreamBuf::TGTextViewStreamBuf(TGTextView *textview) :
   fTextView(textview)
{
   // TGTextViewStreamBuf constructor.

   fInputbuffer.reserve(32);
   setg(&fInputbuffer[0], &fInputbuffer[0], &fInputbuffer[0]);
   setp(&fInputbuffer[0], &fInputbuffer[0]);
}

//______________________________________________________________________________
Int_t TGTextViewStreamBuf::overflow(Int_t c)
{
   // Method called to put a character into the controlled output sequence 
   // without changing the current position.

   typedef std::char_traits<char> Tr;
   if (c == Tr::eof())
      return Tr::not_eof(c);
   if (c=='\n') {
      fLinebuffer.push_back('\0');
      fTextView->AddLineFast(&fLinebuffer[0]);
      fLinebuffer.clear();
      fTextView->ShowBottom();
      fTextView->Update();
      gSystem->ProcessEvents();
   } else {
      fLinebuffer.push_back(c);
   }
   return c;
}

//______________________________________________________________________________
TGTextViewostream::TGTextViewostream(const TGWindow* parent, UInt_t w,
                                     UInt_t h,Int_t id, UInt_t sboptions,
                                     Pixel_t back) :
   TGTextView(parent, w, h, id, sboptions, back), std::ostream(&fStreambuffer),
   fStreambuffer(this)
{
   // TGTextViewostream constructor.

}

//______________________________________________________________________________
TGTextViewostream::TGTextViewostream(const TGWindow *parent, UInt_t w,
                                     UInt_t h, TGText *text, Int_t id,
                                     UInt_t sboptions, ULong_t back):
   TGTextView(parent, w, h, text, id, sboptions, back),
   std::ostream(&fStreambuffer), fStreambuffer(this)
{
   // TGTextViewostream constructor.

}

//______________________________________________________________________________
TGTextViewostream::TGTextViewostream(const TGWindow *parent, UInt_t w,
                                     UInt_t h,const char *string, Int_t id,
                                     UInt_t sboptions, ULong_t back):
   TGTextView(parent, w, h, string, id, sboptions, back),
   std::ostream(&fStreambuffer), fStreambuffer(this)
{
   // TGTextViewostream constructor.

}

