// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   26/04/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGText
#define ROOT_TGText


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGText                                                               //
//                                                                      //
// A TGText is a multi line text buffer. It allows the text to be       //
// loaded from file, saved to file and edited. It is used in the        //
// TGTextEdit widget. Single line text is handled by TGTextBuffer.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObjString
#include "TObjString.h"
#endif

#ifndef ROOT_TGDimension
#include "TGDimension.h"
#endif

class TList;


class TGText {

protected:
   Bool_t       fIsSaved;        // false if text needs to be saved
   TList       *fLines;          // list of lines
   TObjString  *fCurrent;        // current line
   Int_t        fCurrentRow;     // current row number
   Int_t        fRowCount;       // number of rows
   Int_t        fColCount;       // number of columns in current line
   Int_t        fLongestLine;    // length of longest line

   void     TGTextP();
   Bool_t   SetCurrentRow(Int_t row);
   void     LongestLine();
   Int_t    UpSearchBM(const TString &s, const char *searchPattern, Bool_t cs);
   Int_t    DownSearchBM(const TString &s, const char *searchPattern, Bool_t cs);

public:
   TGText();
   TGText(TGText *text);
   TGText(const char *string);
   virtual ~TGText();

   void    Clear();
   Bool_t  Load(const char *fn, Int_t startpos = 0, Int_t length = -1);
   Bool_t  Save(const char *fn);
   Bool_t  Append(const char *fn);
   Bool_t  IsSaved() const { return fIsSaved; }

   Bool_t  DelChar(TGPosition pos);
   Bool_t  InsChar(TGPosition pos, char c);
   char    GetChar(TGPosition pos);

   Bool_t  DelText(TGPosition start, TGPosition end);
   Bool_t  InsText(TGPosition ins_pos, TGText *src, TGPosition start_src, TGPosition end_src);

   Bool_t  DelLine(Int_t row);
   Bool_t  InsLine(TGPosition pos, const TString &str);
   Bool_t  GetLine(TGPosition pos, TString &str);
   Bool_t  BreakLine(TGPosition pos);

   Int_t   RowCount() const { return fRowCount; }
   Int_t   ColCount() const { return fColCount; }
   Int_t   GetLineLength(Int_t row);
   Int_t   GetLongestLine() const { return fLongestLine; }
   Bool_t  Search(TGPosition *foundPos, TGPosition start, const char *searchString,
                  Bool_t direction, Bool_t caseSensitive);
   Bool_t  Replace(TGPosition pos, const char *oldText, const char *newText,
                   Bool_t direction, Bool_t caseSensitive);

   ClassDef(TGText,0)  // Text used by TGTextEdit
};

#endif
