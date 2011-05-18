// @(#)root/gui:$Id$
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

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TGDimension
#include "TGDimension.h"
#endif


class TGTextLine {

friend class TGText;

protected:
   char         *fString;   // line of text
   ULong_t       fLength;   // lenght of line
   TGTextLine   *fPrev;     // previous line
   TGTextLine   *fNext;     // next line

   TGTextLine(const TGTextLine&);
   TGTextLine& operator=(const TGTextLine&);

public:
   TGTextLine();
   TGTextLine(TGTextLine *line);
   TGTextLine(const char *string);
   virtual ~TGTextLine();

   void Clear();
   ULong_t GetLineLength() { return fLength; }

   void DelText(ULong_t pos, ULong_t length);
   void InsText(ULong_t pos, const char *text);
   char *GetText(ULong_t pos, ULong_t length);
   char *GetText() const { return fString; }
   char *GetWord(ULong_t pos);

   void DelChar(ULong_t pos);
   void InsChar(ULong_t pos, char character);
   char GetChar(ULong_t pos);

   ClassDef(TGTextLine,0)  // Line in TGText
};


class TGText {

protected:
   TString      fFilename;       // name of opened file ("" if open buffer)
   Bool_t       fIsSaved;        // false if text needs to be saved
   TGTextLine  *fFirst;          // first line of text
   TGTextLine  *fCurrent;        // current line
   Long_t       fCurrentRow;     // current row number
   Long_t       fRowCount;       // number of rows
   Long_t       fColCount;       // number of columns in current line
   Long_t       fLongestLine;    // length of longest line

   TGText(const TGText&);
   TGText& operator=(const TGText&);

   void     Init();
   Bool_t   SetCurrentRow(Long_t row);
   void     LongestLine();

public:
   TGText();
   TGText(TGText *text);
   TGText(const char *string);
   virtual ~TGText();

   void    Clear();
   Bool_t  Load(const char *fn, Long_t startpos = 0, Long_t length = -1);
   Bool_t  LoadBuffer(const char *txtbuf);
   Bool_t  Save(const char *fn);
   Bool_t  Append(const char *fn);
   Bool_t  IsSaved() const { return fIsSaved; }
   const char *GetFileName() const { return fFilename.Data(); }

   Bool_t  DelChar(TGLongPosition pos);
   Bool_t  InsChar(TGLongPosition pos, char c);
   char    GetChar(TGLongPosition pos);

   Bool_t  DelText(TGLongPosition start, TGLongPosition end);
   Bool_t  InsText(TGLongPosition pos, const char *buf);
   Bool_t  InsText(TGLongPosition ins_pos, TGText *src, TGLongPosition start_src, TGLongPosition end_src);
   Bool_t  AddText(TGText *text);

   Bool_t  DelLine(ULong_t pos);
   char   *GetLine(TGLongPosition pos, ULong_t length);
   TString AsString();
   TGTextLine *GetCurrentLine() const { return fCurrent; }
   Bool_t  BreakLine(TGLongPosition pos);
   Bool_t  InsLine(ULong_t row, const char *string);

   Long_t  RowCount() const { return fRowCount; }
   Long_t  ColCount() const { return fColCount; }

   Long_t  GetLineLength(Long_t row);
   Long_t  GetLongestLine() const { return fLongestLine; }

   void    ReTab(Long_t row);

   Bool_t  Search(TGLongPosition *foundPos, TGLongPosition start, const char *searchString,
                  Bool_t direction, Bool_t caseSensitive);
   Bool_t  Replace(TGLongPosition start, const char *oldText, const char *newText,
                   Bool_t direction, Bool_t caseSensitive);

   ClassDef(TGText,0)  // Text used by TGTextEdit
};

#endif
