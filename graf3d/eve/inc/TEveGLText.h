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

struct TexGlyphInfo_t
{
   short unsigned fC;       // Potentially support 16-bit glyphs.
   unsigned char  fWidth;
   unsigned char  fHeight;
   signed char    fXoffset;
   signed char    fYoffset;
   signed char    fAdvance;
   char           fDummy;   // Space holder for alignment reasons.
   short          fX;
   short          fY;
};

struct TexGlyphVertexInfo_t
{
   GLfloat fT0[2];
   GLshort fV0[2];
   GLfloat fT1[2];
   GLshort fV1[2];
   GLfloat fT2[2];
   GLshort fV2[2];
   GLfloat fT3[2];
   GLshort fV3[2];
   GLfloat fAdvance;
};

struct TexFont_t
{
   GLuint                 fTexObj;
   int                    fTexWidth;
   int                    fTexHeight;
   int                    fMaxAscent;
   int                    fMaxDescent;
   int                    fMaxWidth;   // max glyph width (MT)
   int                    fNumGlyphs;
   int                    fMinGlyph;
   int                    fRange;
   unsigned char         *fTeximage;
   TexGlyphInfo_t        *fTgi;
   TexGlyphVertexInfo_t  *fTgvi;
   TexGlyphVertexInfo_t **fLut;

   int MaxHeight() { return fMaxAscent + fMaxDescent; }
};


extern const char *txfErrorString(void);

extern TexFont_t *txfLoadFont(const char *filename);

extern void txfUnloadFont(TexFont_t* txf);

extern GLuint txfEstablishTexture(TexFont_t* txf, GLuint texobj,
                                  GLboolean setupMipmaps);

extern void txfBindFontTexture(TexFont_t* txf);

extern void txfGetStringMetrics(TexFont_t* txf, const char *TString, int len,
                                int &width, int &max_ascent, int &max_descent);

extern void txfRenderGlyph(TexFont_t* txf, int c);
extern void txfRenderString(TexFont_t* txf, const char *TString, int len,
                            bool keep_pos=true);
extern void txfRenderString(TexFont_t* txf, const char *TString, int len,
                            GLfloat maxx, GLfloat fadew,
                            bool keep_pos=true);

extern void txfRenderGlyphZW(TexFont_t* txf, int c, float z, float w);
extern void txfRenderStringZW(TexFont_t* txf, const char *TString, int len,
                              float z, float w, bool keep_pos=true);

extern void txfRenderFancyString(TexFont_t* txf, char *TString, int len);


bool        LoadDefaultFont(const TString& font_file);

extern TexFont_t* fgDefaultFont;

}  // namescape TEveGLText

#endif  // cint
#endif  // Reve_GLTextNS_H
