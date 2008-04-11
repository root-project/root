// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// The following implementation is based on TexFont API,
// implementation and accompanying programs by Mark J. Kilgard.
// Original license:

/* Copyright (c) Mark J. Kilgard, 1997. */
/* This program is freely distributable without licensing fees  and is
   provided without guarantee or warrantee expressed or implied. This
   program is -not- in the public domain. */

#include "TEveGLText.h"
#include "TEveUtil.h"

#include "TGLIncludes.h"
#include "TGLUtil.h"

#include "TMath.h"
#include "TString.h"

#include <cassert>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>


/**********************/
namespace TEveGLText {
/**********************/

TexFont_t* fgDefaultFont = 0;

#if 0
/* Uncomment to debug various scenarios. */
#undef GL_VERSION_1_1
#undef GL_EXT_texture_object
#undef GL_EXT_texture
#endif

int useLuminanceAlpha = 1;

/* byte swap a 32-bit value */
#define SWAPL(x, n) {                           \
      n = ((char *) (x))[0];                    \
      ((char *) (x))[0] = ((char *) (x))[3];    \
      ((char *) (x))[3] = n;                    \
      n = ((char *) (x))[1];                    \
      ((char *) (x))[1] = ((char *) (x))[2];    \
      ((char *) (x))[2] = n; }

/* byte swap a short */
#define SWAPS(x, n) {                           \
      n = ((char *) (x))[0];                    \
      ((char *) (x))[0] = ((char *) (x))[1];    \
      ((char *) (x))[1] = n; }

/******************************************************************************/

//______________________________________________________________________________
static TexGlyphVertexInfo_t* getTCVI(TexFont_t * txf, int c)
{
   TexGlyphVertexInfo_t *tgvi;

   /* Automatically substitute uppercase letters with lowercase if not
      uppercase available (and vice versa). */
   if ((c >= txf->fMinGlyph) && (c < txf->fMinGlyph + txf->fRange)) {
      tgvi = txf->fLut[c - txf->fMinGlyph];
      if (tgvi) {
         return tgvi;
      }
      if (islower(c)) {
         c = toupper(c);
         if ((c >= txf->fMinGlyph) && (c < txf->fMinGlyph + txf->fRange)) {
            return txf->fLut[c - txf->fMinGlyph];
         }
      }
      if (isupper(c)) {
         c = tolower(c);
         if ((c >= txf->fMinGlyph) && (c < txf->fMinGlyph + txf->fRange)) {
            return txf->fLut[c - txf->fMinGlyph];
         }
      }
   }

   //fprintf(stderr, "texfont: tried to access unavailable font character \"%c\" (%d)\n",
   //    isprint(c) ? c : ' ', c);

   tgvi = txf->fLut[' ' - txf->fMinGlyph];
   if (tgvi) return tgvi;
   tgvi = txf->fLut['_' - txf->fMinGlyph];
   if (tgvi) return tgvi;

   return 0;
}

/******************************************************************************/

static const char *lastError;

//______________________________________________________________________________
const char* txfErrorString(void)
{
   return lastError;
}

/******************************************************************************/

//______________________________________________________________________________
TexFont_t* txfLoadFont(const char *filename)
{
   TexFont_t *txf;
   FILE *file;
   GLfloat w, h, xstep, ystep;
   char fileid[4], tmp;
   unsigned char *texbitmap;
   int min_glyph, max_glyph;
   int endianness, swap, format, stride, width, height;
   int i, j, got;

   txf = 0;
   file = fopen(filename, "rb");
   if (file == 0) {
      lastError = "file open failed.";
      goto error;
   }
   txf = (TexFont_t *) malloc(sizeof(TexFont_t));
   if (txf == 0) {
      lastError = "out of memory.";
      goto error;
   }
   /* For easy cleanup in error case. */
   txf->fTexObj = 0; // MT add
   txf->fTgi = 0;
   txf->fTgvi = 0;
   txf->fLut = 0;
   txf->fTeximage = 0;

   got = fread(fileid, 1, 4, file);
   if (got != 4 || strncmp(fileid, "\377txf", 4)) {
      lastError = "not a texture font file.";
      goto error;
   }
   assert(sizeof(int) == 4);  /* Ensure external file format size. */
   got = fread(&endianness, sizeof(int), 1, file);
   if (got == 1 && endianness == 0x12345678) {
      swap = 0;
   } else if (got == 1 && endianness == 0x78563412) {
      swap = 1;
   } else {
      lastError = "not a texture font file.";
      goto error;
   }
#define EXPECT(n) if (got != n) { lastError = "premature end of file."; goto error; }
   got = fread(&format, sizeof(int), 1, file);
   EXPECT(1);
   got = fread(&txf->fTexWidth, sizeof(int), 1, file);
   EXPECT(1);
   got = fread(&txf->fTexHeight, sizeof(int), 1, file);
   EXPECT(1);
   got = fread(&txf->fMaxAscent, sizeof(int), 1, file);
   EXPECT(1);
   got = fread(&txf->fMaxDescent, sizeof(int), 1, file);
   EXPECT(1);
   got = fread(&txf->fNumGlyphs, sizeof(int), 1, file);
   EXPECT(1);

   if (swap) {
      SWAPL(&format, tmp);
      SWAPL(&txf->fTexWidth, tmp);
      SWAPL(&txf->fTexHeight, tmp);
      SWAPL(&txf->fMaxAscent, tmp);
      SWAPL(&txf->fMaxDescent, tmp);
      SWAPL(&txf->fNumGlyphs, tmp);
   }
   txf->fTgi = (TexGlyphInfo_t *) malloc(txf->fNumGlyphs * sizeof(TexGlyphInfo_t));
   if (txf->fTgi == 0) {
      lastError = "out of memory.";
      goto error;
   }
   assert(sizeof(TexGlyphInfo_t) == 12);  /* Ensure external file format size. */
   got = fread(txf->fTgi, sizeof(TexGlyphInfo_t), txf->fNumGlyphs, file);
   EXPECT(txf->fNumGlyphs);

   if (swap) {
      for (i = 0; i < txf->fNumGlyphs; i++) {
         SWAPS(&txf->fTgi[i].fC, tmp);
         SWAPS(&txf->fTgi[i].fX, tmp);
         SWAPS(&txf->fTgi[i].fY, tmp);
      }
   }
   txf->fTgvi = (TexGlyphVertexInfo_t *)
      malloc(txf->fNumGlyphs * sizeof(TexGlyphVertexInfo_t));
   if (txf->fTgvi == 0) {
      lastError = "out of memory.";
      goto error;
   }
   w = txf->fTexWidth;
   h = txf->fTexHeight;
   txf->fMaxWidth = 0;
   xstep = 0.5 / w;
   ystep = 0.5 / h;
   for (i = 0; i < txf->fNumGlyphs; i++) {
      TexGlyphInfo_t *tgi;

      tgi = &txf->fTgi[i];
      txf->fTgvi[i].fT0[0] = tgi->fX / w - xstep; // MT - xstep
      txf->fTgvi[i].fT0[1] = tgi->fY / h - ystep; // MT - ystep
      txf->fTgvi[i].fV0[0] = tgi->fXoffset;
      txf->fTgvi[i].fV0[1] = tgi->fYoffset;
      txf->fTgvi[i].fT1[0] = (tgi->fX + tgi->fWidth) / w + xstep;
      txf->fTgvi[i].fT1[1] = tgi->fY / h - ystep; // MT - ystep
      txf->fTgvi[i].fV1[0] = tgi->fXoffset + tgi->fWidth;
      txf->fTgvi[i].fV1[1] = tgi->fYoffset;
      txf->fTgvi[i].fT2[0] = (tgi->fX + tgi->fWidth) / w + xstep;
      txf->fTgvi[i].fT2[1] = (tgi->fY + tgi->fHeight) / h + ystep;
      txf->fTgvi[i].fV2[0] = tgi->fXoffset + tgi->fWidth;
      txf->fTgvi[i].fV2[1] = tgi->fYoffset + tgi->fHeight;
      txf->fTgvi[i].fT3[0] = tgi->fX / w - xstep; // MT - xstep
      txf->fTgvi[i].fT3[1] = (tgi->fY + tgi->fHeight) / h + ystep;
      txf->fTgvi[i].fV3[0] = tgi->fXoffset;
      txf->fTgvi[i].fV3[1] = tgi->fYoffset + tgi->fHeight;
      txf->fTgvi[i].fAdvance = tgi->fAdvance;

      if(tgi->fWidth > txf->fMaxWidth) txf->fMaxWidth = tgi->fWidth;
   }

   min_glyph = txf->fTgi[0].fC;
   max_glyph = txf->fTgi[0].fC;
   for (i = 1; i < txf->fNumGlyphs; i++) {
      if (txf->fTgi[i].fC < min_glyph) {
         min_glyph = txf->fTgi[i].fC;
      }
      if (txf->fTgi[i].fC > max_glyph) {
         max_glyph = txf->fTgi[i].fC;
      }
   }
   txf->fMinGlyph = min_glyph;
   txf->fRange = max_glyph - min_glyph + 1;

   txf->fLut = (TexGlyphVertexInfo_t **)
      calloc(txf->fRange, sizeof(TexGlyphVertexInfo_t *));
   if (txf->fLut == 0) {
      lastError = "out of memory.";
      goto error;
   }
   for (i = 0; i < txf->fNumGlyphs; i++) {
      txf->fLut[txf->fTgi[i].fC - txf->fMinGlyph] = &txf->fTgvi[i];
   }

   switch (format) {
      case TXF_FORMAT_BYTE:
         if (useLuminanceAlpha) {
            unsigned char *orig;

            orig = (unsigned char *) malloc(txf->fTexWidth * txf->fTexHeight);
            if (orig == 0) {
               lastError = "out of memory.";
               goto error;
            }
            got = fread(orig, 1, txf->fTexWidth * txf->fTexHeight, file);
            EXPECT(txf->fTexWidth * txf->fTexHeight);
            txf->fTeximage = (unsigned char *)
               malloc(2 * txf->fTexWidth * txf->fTexHeight);
            if (txf->fTeximage == 0) {
               lastError = "out of memory.";
               goto error;
            }
            for (i = 0; i < txf->fTexWidth * txf->fTexHeight; i++) {
               txf->fTeximage[i * 2] = orig[i];
               txf->fTeximage[i * 2 + 1] = orig[i];
            }
            free(orig);
         } else {
            txf->fTeximage = (unsigned char *)
               malloc(txf->fTexWidth * txf->fTexHeight);
            if (txf->fTeximage == 0) {
               lastError = "out of memory.";
               goto error;
            }
            got = fread(txf->fTeximage, 1, txf->fTexWidth * txf->fTexHeight, file);
            EXPECT(txf->fTexWidth * txf->fTexHeight);
         }
         break;
      case TXF_FORMAT_BITMAP:
         width = txf->fTexWidth;
         height = txf->fTexHeight;
         stride = (width + 7) >> 3;
         texbitmap = (unsigned char *) malloc(stride * height);
         if (texbitmap == 0) {
            lastError = "out of memory.";
            goto error;
         }
         got = fread(texbitmap, 1, stride * height, file);
         EXPECT(stride * height);
         if (useLuminanceAlpha) {
            txf->fTeximage = (unsigned char *) calloc(width * height * 2, 1);
            if (txf->fTeximage == 0) {
               lastError = "out of memory.";
               goto error;
            }
            for (i = 0; i < height; i++) {
               for (j = 0; j < width; j++) {
                  if (texbitmap[i * stride + (j >> 3)] & (1 << (j & 7))) {
                     txf->fTeximage[(i * width + j) * 2] = 255;
                     txf->fTeximage[(i * width + j) * 2 + 1] = 255;
                  }
               }
            }
         } else {
            txf->fTeximage = (unsigned char *) calloc(width * height, 1);
            if (txf->fTeximage == 0) {
               lastError = "out of memory.";
               goto error;
            }
            for (i = 0; i < height; i++) {
               for (j = 0; j < width; j++) {
                  if (texbitmap[i * stride + (j >> 3)] & (1 << (j & 7))) {
                     txf->fTeximage[i * width + j] = 255;
                  }
               }
            }
         }
         free(texbitmap);
         break;
   }

   fclose(file);
   return txf;

error:

   if (txf) {
      if (txf->fTgi)
         free(txf->fTgi);
      if (txf->fTgvi)
         free(txf->fTgvi);
      if (txf->fLut)
         free(txf->fLut);
      if (txf->fTeximage)
         free(txf->fTeximage);
      free(txf);
   }
   if (file)
      fclose(file);
   return 0;
}

/******************************************************************************/

//______________________________________________________________________________
GLuint txfEstablishTexture(TexFont_t * txf, GLuint texobj,
                           GLboolean setupMipmaps)
{
   if (txf->fTexObj == 0) {
      if (texobj == 0) {
         glGenTextures(1, &txf->fTexObj);
      } else {
         txf->fTexObj = texobj;
      }
   }
   glBindTexture(GL_TEXTURE_2D, txf->fTexObj);

   if (useLuminanceAlpha) {
      if (setupMipmaps) {
         gluBuild2DMipmaps(GL_TEXTURE_2D, GL_LUMINANCE_ALPHA,
                           txf->fTexWidth, txf->fTexHeight,
                           GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, txf->fTeximage);
      } else {
         glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE_ALPHA,
                      txf->fTexWidth, txf->fTexHeight, 0,
                      GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, txf->fTeximage);
      }
   } else {
      if (setupMipmaps) {
         gluBuild2DMipmaps(GL_TEXTURE_2D, GL_INTENSITY4,
                           txf->fTexWidth, txf->fTexHeight,
                           GL_LUMINANCE, GL_UNSIGNED_BYTE, txf->fTeximage);
      } else {
         glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY4,
                      txf->fTexWidth, txf->fTexHeight, 0,
                      GL_LUMINANCE, GL_UNSIGNED_BYTE, txf->fTeximage);
      }
   }

   // MT: tried changing MIN/MAG filters ... bad idea.

   return txf->fTexObj;
}

/******************************************************************************/

//______________________________________________________________________________
void txfBindFontTexture(TexFont_t * txf)
{
   glBindTexture(GL_TEXTURE_2D, txf->fTexObj);
}

/******************************************************************************/

//______________________________________________________________________________
void txfUnloadFont(TexFont_t * txf)
{
   if (txf->fTexObj) {
      glDeleteTextures(1, &txf->fTexObj);
   }
   if (txf->fTeximage) {
      free(txf->fTeximage);
   }
   free(txf->fTgi);
   free(txf->fTgvi);
   free(txf->fLut);
   free(txf);
}

/******************************************************************************/

//______________________________________________________________________________
void txfGetStringMetrics(TexFont_t * txf, const char *TString, int len,
                         int &width, int &max_ascent, int &max_descent)
{
   TexGlyphVertexInfo_t *tgvi;
   int     w, i;
   int ma = 0, md = 0;

   w = 0;
   for (i = 0; i < len; i++) {
      if (TString[i] == 27) {
         switch (TString[i + 1]) {
            case 'M':
               i += 4;
               break;
            case 'T':
               i += 7;
               break;
            case 'L':
               i += 7;
               break;
            case 'F':
               i += 13;
               break;
         }
      } else {
         tgvi = getTCVI(txf, TString[i]);
         w += int(tgvi->fAdvance);
         ma = TMath::Max(ma, (int)( tgvi->fV3[1]));
         md = TMath::Max(md, (int)(-tgvi->fV0[1]));
      }
   }
   width = w;
   max_ascent  = ma; // txf->fMaxAscent;
   max_descent = md; // txf->fMaxDescent;
   // printf("%d %d %d %d\n", txf->fMaxAscent, txf->fMaxDescent, ma, md);
}

/******************************************************************************/

//______________________________________________________________________________
void txfRenderGlyph(TexFont_t * txf, int c)
{
   TexGlyphVertexInfo_t *tgvi;

   tgvi = getTCVI(txf, c);
   glBegin(GL_QUADS);
   glTexCoord2fv(tgvi->fT0);
   glVertex2sv(tgvi->fV0);
   glTexCoord2fv(tgvi->fT1);
   glVertex2sv(tgvi->fV1);
   glTexCoord2fv(tgvi->fT2);
   glVertex2sv(tgvi->fV2);
   glTexCoord2fv(tgvi->fT3);
   glVertex2sv(tgvi->fV3);
   glEnd();
   glTranslatef(tgvi->fAdvance, 0.0, 0.0);
}

//______________________________________________________________________________
void txfRenderString(TexFont_t * txf, const char *TString, int len,
                     bool keep_pos)
{
   int i;
   if(keep_pos) glPushMatrix();
   for (i = 0; i < len; i++) {
      txfRenderGlyph(txf, TString[i]);
   }
   if(keep_pos) glPopMatrix();
}

//______________________________________________________________________________
void txfRenderString(TexFont_t * txf, const char *TString, int len,
                     GLfloat maxx, GLfloat fadew,
                     bool keep_pos)
{
   GLfloat x = 0, xg0, xg1, yg0, yg1, f0, f1;
   fadew *= txf->fMaxWidth;
   GLfloat xfade = maxx - fadew;

   GLfloat col[4];
   glGetFloatv(GL_CURRENT_COLOR, col);

   glBegin(GL_QUADS);
   for (int i = 0; i < len; i++) {

      TexGlyphVertexInfo_t *tgvi;

      tgvi = getTCVI(txf, TString[i]);

      xg0 = x + tgvi->fV0[0];
      xg1 = x + tgvi->fV1[0];
      yg0 = tgvi->fV0[1];
      yg1 = tgvi->fV2[1];

      if(xg1 > xfade) {
         f0 = 1;        if(xg0 > xfade) f0 *= 1 - (xg0-xfade)/fadew;
         f1 = 1 - (xg1-xfade)/fadew;

         // printf("XX %s %c %f %f x(%f,%f) y(%f,%f)\n",
         //        TString, TString[i], f0, f1,
         //        xg0, xg1,yg0, yg1);

         TGLUtil::Color4f(f0*col[0], f0*col[1], f0*col[2], f0*col[3]);
         glTexCoord2fv(tgvi->fT0);    glVertex2f(xg0, yg0);
         TGLUtil::Color4f(f1*col[0], f1*col[1], f1*col[2], f1*col[3]);
         glTexCoord2fv(tgvi->fT1);    glVertex2f(xg1, yg0);
         glTexCoord2fv(tgvi->fT2);    glVertex2f(xg1, yg1);
         TGLUtil::Color4f(f0*col[0], f0*col[1], f0*col[2], f0*col[3]);
         glTexCoord2fv(tgvi->fT3);    glVertex2f(xg0, yg1);
      } else {
         glTexCoord2fv(tgvi->fT0);    glVertex2f(xg0, yg0);
         glTexCoord2fv(tgvi->fT1);    glVertex2f(xg1, yg0);
         glTexCoord2fv(tgvi->fT2);    glVertex2f(xg1, yg1);
         glTexCoord2fv(tgvi->fT3);    glVertex2f(xg0, yg1);
      }

      x += tgvi->fAdvance;
      if(x > maxx) break;
   }
   glEnd();

   if(!keep_pos) glTranslatef(x, 0.0, 0.0);
}

/******************************************************************************/

//______________________________________________________________________________
void txfRenderGlyphZW(TexFont_t * txf, int c, float z, float w)
{
   TexGlyphVertexInfo_t *tgvi;

   tgvi = getTCVI(txf, c);
   glBegin(GL_QUADS);
   glTexCoord2fv(tgvi->fT0);
   glVertex4f(tgvi->fV0[0], tgvi->fV0[1], z, w);
   glTexCoord2fv(tgvi->fT1);
   glVertex4f(tgvi->fV1[0], tgvi->fV1[1], z, w);
   glTexCoord2fv(tgvi->fT2);
   glVertex4f(tgvi->fV2[0], tgvi->fV2[1], z, w);
   glTexCoord2fv(tgvi->fT3);
   glVertex4f(tgvi->fV3[0], tgvi->fV3[1], z, w);
   glEnd();
   glTranslatef(tgvi->fAdvance, 0.0, 0.0);
}

//______________________________________________________________________________
void txfRenderStringZW(TexFont_t * txf, const char *TString, int len,
                       float z, float w, bool keep_pos)
{
   int i;

   if(keep_pos) glPushMatrix();
   for (i = 0; i < len; i++) {
      txfRenderGlyphZW(txf, TString[i], z, w);
   }
   if(keep_pos) glPopMatrix();
}

/******************************************************************************/

enum {
   kMONO, kTOP_BOTTOM, kLEFT_RIGHT, kFOUR
};

/******************************************************************************/

//______________________________________________________________________________
void txfRenderFancyString(TexFont_t * txf, char *TString, int len)
{
   TexGlyphVertexInfo_t *tgvi;
   GLubyte c[4][3];
   int mode = kMONO;
   int i;

   for (i = 0; i < len; i++) {
      if (TString[i] == 27) {
         switch (TString[i + 1]) {
            case 'M':
               mode = kMONO;
               TGLUtil::Color3ubv((GLubyte *) & TString[i + 2]);
               i += 4;
               break;
            case 'T':
               mode = kTOP_BOTTOM;
               memcpy(c, &TString[i + 2], 6);
               i += 7;
               break;
            case 'L':
               mode = kLEFT_RIGHT;
               memcpy(c, &TString[i + 2], 6);
               i += 7;
               break;
            case 'F':
               mode = kFOUR;
               memcpy(c, &TString[i + 2], 12);
               i += 13;
               break;
         }
      } else {
         switch (mode) {
            case kMONO:
               txfRenderGlyph(txf, TString[i]);
               break;
            case kTOP_BOTTOM:
               tgvi = getTCVI(txf, TString[i]);
               glBegin(GL_QUADS);
               TGLUtil::Color3ubv(c[0]);
               glTexCoord2fv(tgvi->fT0);
               glVertex2sv(tgvi->fV0);
               glTexCoord2fv(tgvi->fT1);
               glVertex2sv(tgvi->fV1);
               TGLUtil::Color3ubv(c[1]);
               glTexCoord2fv(tgvi->fT2);
               glVertex2sv(tgvi->fV2);
               glTexCoord2fv(tgvi->fT3);
               glVertex2sv(tgvi->fV3);
               glEnd();
               glTranslatef(tgvi->fAdvance, 0.0, 0.0);
               break;
            case kLEFT_RIGHT:
               tgvi = getTCVI(txf, TString[i]);
               glBegin(GL_QUADS);
               TGLUtil::Color3ubv(c[0]);
               glTexCoord2fv(tgvi->fT0);
               glVertex2sv(tgvi->fV0);
               TGLUtil::Color3ubv(c[1]);
               glTexCoord2fv(tgvi->fT1);
               glVertex2sv(tgvi->fV1);
               TGLUtil::Color3ubv(c[1]);
               glTexCoord2fv(tgvi->fT2);
               glVertex2sv(tgvi->fV2);
               TGLUtil::Color3ubv(c[0]);
               glTexCoord2fv(tgvi->fT3);
               glVertex2sv(tgvi->fV3);
               glEnd();
               glTranslatef(tgvi->fAdvance, 0.0, 0.0);
               break;
            case kFOUR:
               tgvi = getTCVI(txf, TString[i]);
               glBegin(GL_QUADS);
               TGLUtil::Color3ubv(c[0]);
               glTexCoord2fv(tgvi->fT0);
               glVertex2sv(tgvi->fV0);
               TGLUtil::Color3ubv(c[1]);
               glTexCoord2fv(tgvi->fT1);
               glVertex2sv(tgvi->fV1);
               TGLUtil::Color3ubv(c[2]);
               glTexCoord2fv(tgvi->fT2);
               glVertex2sv(tgvi->fV2);
               TGLUtil::Color3ubv(c[3]);
               glTexCoord2fv(tgvi->fT3);
               glVertex2sv(tgvi->fV3);
               glEnd();
               glTranslatef(tgvi->fAdvance, 0.0, 0.0);
               break;
         }
      }
   }
}

/******************************************************************************/

//______________________________________________________________________________
int txfInFont(TexFont_t* txf, int c)
{
   /* NOTE: No uppercase/lowercase substituion. */
   if ((c >= txf->fMinGlyph) && (c < txf->fMinGlyph + txf->fRange)) {
      if (txf->fLut[c - txf->fMinGlyph]) {
         return 1;
      }
   }
   return 0;
}

/******************************************************************************/

//______________________________________________________________________________
bool LoadDefaultFont(const TString& file)
{
   static const TEveException eH("TEveGLText::LoadFont ");

   if(fgDefaultFont) {
      txfUnloadFont(fgDefaultFont);
      fgDefaultFont = 0;
   }

   fgDefaultFont = TEveGLText::txfLoadFont(file.Data());
   if(fgDefaultFont != 0) {
      txfEstablishTexture(fgDefaultFont, 0, GL_TRUE);
      return true;
   }
   else {
      throw(eH + Form("Error loading font from file '%s': %s",
                      file.Data(), txfErrorString()));
   }

   return false;
}

} // end TEveGLText
