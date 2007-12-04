// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGLText
#define ROOT_TEveGLText

// The following implementation is based on TexFont API,
// implementation and accompanying programs by Mark J. Kilgard.
// Original license:

/* Copyright (c) Mark J. Kilgard, 1997. */
/* This program is freely distributable without licensing fees  and is
   provided without guarantee or warrantee expressed or implied. This
   program is -not- in the public domain. */

#ifndef __CINT__

#include "TObject.h"
#include "TGLIncludes.h"

class TString;

namespace TEveGLText {

#define TXF_FORMAT_BYTE          0
#define TXF_FORMAT_BITMAP        1

struct TexGlyphInfo_t {
   unsigned short c;       /* Potentially support 16-bit glyphs. */
   unsigned char width;
   unsigned char height;
   signed char xoffset;
   signed char yoffset;
   signed char advance;
   char dummy;           /* Space holder for alignment reasons. */
   short x;
   short y;
};

struct TexGlyphVertexInfo_t {
   GLfloat t0[2];
   GLshort v0[2];
   GLfloat t1[2];
   GLshort v1[2];
   GLfloat t2[2];
   GLshort v2[2];
   GLfloat t3[2];
   GLshort v3[2];
   GLfloat advance;
};

class TexFont : public TObject {
public:
   GLuint texobj;
   int tex_width;
   int tex_height;
   int max_ascent;
   int max_descent;
   int max_width;   // max glyph width (MT)
   int num_glyphs;
   int min_glyph;
   int range;
   unsigned char *teximage;
   TexGlyphInfo_t *tgi;
   TexGlyphVertexInfo_t *tgvi;
   TexGlyphVertexInfo_t **lut;

   int max_height() { return max_ascent + max_descent; }
};


extern const char *txfErrorString(void);

extern TexFont *txfLoadFont(const char *filename);

extern void txfUnloadFont(TexFont* txf);

extern GLuint txfEstablishTexture(TexFont* txf, GLuint texobj,
                                  GLboolean setupMipmaps);

extern void txfBindFontTexture(TexFont* txf);

extern void txfGetStringMetrics(TexFont* txf, const char *TString, int len,
                                int &width, int &max_ascent, int &max_descent);

extern void txfRenderGlyph(TexFont* txf, int c);
extern void txfRenderString(TexFont* txf, const char *TString, int len,
                            bool keep_pos=true);
extern void txfRenderString(TexFont* txf, const char *TString, int len,
                            GLfloat maxx, GLfloat fadew,
                            bool keep_pos=true);

extern void txfRenderGlyphZW(TexFont* txf, int c, float z, float w);
extern void txfRenderStringZW(TexFont* txf, const char *TString, int len,
                              float z, float w, bool keep_pos=true);

extern void txfRenderFancyString(TexFont* txf, char *TString, int len);


bool        LoadDefaultFont(const TString& font_file);

extern TexFont* fgDefaultFont;

/******************************************************************************/
// Here starts MT higher-level interface
/******************************************************************************/
/*
  struct BoxSpecs {
  int     lm, rm, tm, bm;
  int     lineskip;
  char    align;
  TString pos;

  void _init() { align = 'l'; lineskip = 0; }

  BoxSpecs()
  { lm = rm = 3; tm = 0; bm = 2; _init(); }

  BoxSpecs(int lr, int tb)
  { lm = rm = lr; tm = bm = tb; _init(); }

  BoxSpecs(int l, int r, int t, int b)
  { lm = l; rm = r; tm = t; bm = b; _init(); }
  };
  struct TextLineData {
  int    width, ascent, descent, hfull;
  TString text;

  TextLineData(TexFont *txf, TString line);
  };

  extern void RnrTextBar(RnrDriver* rd, const TString& text);

  extern void RnrTextBar(RnrDriver* rd, const TString& text,
  BoxSpecs& bs, float zoffset=0);

  extern void RnrTextPoly(RnrDriver* rd, const TString& text);

  extern void RnrText(RnrDriver* rd, const TString& text,
  int x, int y, float z,
  const ZColor* front_col, const ZColor* back_col=0);

  extern void RnrTextAt(RnrDriver* rd, const TString& text,
  int x, int yrow, float z,
  const ZColor* front_col, const ZColor* back_col=0);
*/

}  // namescape TEveGLText

#endif  // cint
#endif  // Reve_GLTextNS_H
