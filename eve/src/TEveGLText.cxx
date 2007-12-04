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

#include "TMath.h"
#include "TString.h"

#include <GL/glu.h>

#include <cassert>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>


/**********************/
namespace TEveGLText {
/**********************/

TexFont* fgDefaultFont = 0;

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
static TexGlyphVertexInfo_t* getTCVI(TexFont * txf, int c)
{
   TexGlyphVertexInfo_t *tgvi;

   /* Automatically substitute uppercase letters with lowercase if not
      uppercase available (and vice versa). */
   if ((c >= txf->min_glyph) && (c < txf->min_glyph + txf->range)) {
      tgvi = txf->lut[c - txf->min_glyph];
      if (tgvi) {
         return tgvi;
      }
      if (islower(c)) {
         c = toupper(c);
         if ((c >= txf->min_glyph) && (c < txf->min_glyph + txf->range)) {
            return txf->lut[c - txf->min_glyph];
         }
      }
      if (isupper(c)) {
         c = tolower(c);
         if ((c >= txf->min_glyph) && (c < txf->min_glyph + txf->range)) {
            return txf->lut[c - txf->min_glyph];
         }
      }
   }

   //fprintf(stderr, "texfont: tried to access unavailable font character \"%c\" (%d)\n",
   //    isprint(c) ? c : ' ', c);

   tgvi = txf->lut[' ' - txf->min_glyph];
   if (tgvi) return tgvi;
   tgvi = txf->lut['_' - txf->min_glyph];
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
TexFont* txfLoadFont(const char *filename)
{
   TexFont *txf;
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
   txf = (TexFont *) malloc(sizeof(TexFont));
   if (txf == 0) {
      lastError = "out of memory.";
      goto error;
   }
   /* For easy cleanup in error case. */
   txf->texobj = 0; // MT add
   txf->tgi = 0;
   txf->tgvi = 0;
   txf->lut = 0;
   txf->teximage = 0;

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
   got = fread(&txf->tex_width, sizeof(int), 1, file);
   EXPECT(1);
   got = fread(&txf->tex_height, sizeof(int), 1, file);
   EXPECT(1);
   got = fread(&txf->max_ascent, sizeof(int), 1, file);
   EXPECT(1);
   got = fread(&txf->max_descent, sizeof(int), 1, file);
   EXPECT(1);
   got = fread(&txf->num_glyphs, sizeof(int), 1, file);
   EXPECT(1);

   if (swap) {
      SWAPL(&format, tmp);
      SWAPL(&txf->tex_width, tmp);
      SWAPL(&txf->tex_height, tmp);
      SWAPL(&txf->max_ascent, tmp);
      SWAPL(&txf->max_descent, tmp);
      SWAPL(&txf->num_glyphs, tmp);
   }
   txf->tgi = (TexGlyphInfo_t *) malloc(txf->num_glyphs * sizeof(TexGlyphInfo_t));
   if (txf->tgi == 0) {
      lastError = "out of memory.";
      goto error;
   }
   assert(sizeof(TexGlyphInfo_t) == 12);  /* Ensure external file format size. */
   got = fread(txf->tgi, sizeof(TexGlyphInfo_t), txf->num_glyphs, file);
   EXPECT(txf->num_glyphs);

   if (swap) {
      for (i = 0; i < txf->num_glyphs; i++) {
         SWAPS(&txf->tgi[i].c, tmp);
         SWAPS(&txf->tgi[i].x, tmp);
         SWAPS(&txf->tgi[i].y, tmp);
      }
   }
   txf->tgvi = (TexGlyphVertexInfo_t *)
      malloc(txf->num_glyphs * sizeof(TexGlyphVertexInfo_t));
   if (txf->tgvi == 0) {
      lastError = "out of memory.";
      goto error;
   }
   w = txf->tex_width;
   h = txf->tex_height;
   txf->max_width = 0;
   xstep = 0.5 / w;
   ystep = 0.5 / h;
   for (i = 0; i < txf->num_glyphs; i++) {
      TexGlyphInfo_t *tgi;

      tgi = &txf->tgi[i];
      txf->tgvi[i].t0[0] = tgi->x / w - xstep; // MT - xstep
      txf->tgvi[i].t0[1] = tgi->y / h - ystep; // MT - ystep
      txf->tgvi[i].v0[0] = tgi->xoffset;
      txf->tgvi[i].v0[1] = tgi->yoffset;
      txf->tgvi[i].t1[0] = (tgi->x + tgi->width) / w + xstep;
      txf->tgvi[i].t1[1] = tgi->y / h - ystep; // MT - ystep
      txf->tgvi[i].v1[0] = tgi->xoffset + tgi->width;
      txf->tgvi[i].v1[1] = tgi->yoffset;
      txf->tgvi[i].t2[0] = (tgi->x + tgi->width) / w + xstep;
      txf->tgvi[i].t2[1] = (tgi->y + tgi->height) / h + ystep;
      txf->tgvi[i].v2[0] = tgi->xoffset + tgi->width;
      txf->tgvi[i].v2[1] = tgi->yoffset + tgi->height;
      txf->tgvi[i].t3[0] = tgi->x / w - xstep; // MT - xstep
      txf->tgvi[i].t3[1] = (tgi->y + tgi->height) / h + ystep;
      txf->tgvi[i].v3[0] = tgi->xoffset;
      txf->tgvi[i].v3[1] = tgi->yoffset + tgi->height;
      txf->tgvi[i].advance = tgi->advance;

      if(tgi->width > txf->max_width) txf->max_width = tgi->width;
   }

   min_glyph = txf->tgi[0].c;
   max_glyph = txf->tgi[0].c;
   for (i = 1; i < txf->num_glyphs; i++) {
      if (txf->tgi[i].c < min_glyph) {
         min_glyph = txf->tgi[i].c;
      }
      if (txf->tgi[i].c > max_glyph) {
         max_glyph = txf->tgi[i].c;
      }
   }
   txf->min_glyph = min_glyph;
   txf->range = max_glyph - min_glyph + 1;

   txf->lut = (TexGlyphVertexInfo_t **)
      calloc(txf->range, sizeof(TexGlyphVertexInfo_t *));
   if (txf->lut == 0) {
      lastError = "out of memory.";
      goto error;
   }
   for (i = 0; i < txf->num_glyphs; i++) {
      txf->lut[txf->tgi[i].c - txf->min_glyph] = &txf->tgvi[i];
   }

   switch (format) {
      case TXF_FORMAT_BYTE:
         if (useLuminanceAlpha) {
            unsigned char *orig;

            orig = (unsigned char *) malloc(txf->tex_width * txf->tex_height);
            if (orig == 0) {
               lastError = "out of memory.";
               goto error;
            }
            got = fread(orig, 1, txf->tex_width * txf->tex_height, file);
            EXPECT(txf->tex_width * txf->tex_height);
            txf->teximage = (unsigned char *)
               malloc(2 * txf->tex_width * txf->tex_height);
            if (txf->teximage == 0) {
               lastError = "out of memory.";
               goto error;
            }
            for (i = 0; i < txf->tex_width * txf->tex_height; i++) {
               txf->teximage[i * 2] = orig[i];
               txf->teximage[i * 2 + 1] = orig[i];
            }
            free(orig);
         } else {
            txf->teximage = (unsigned char *)
               malloc(txf->tex_width * txf->tex_height);
            if (txf->teximage == 0) {
               lastError = "out of memory.";
               goto error;
            }
            got = fread(txf->teximage, 1, txf->tex_width * txf->tex_height, file);
            EXPECT(txf->tex_width * txf->tex_height);
         }
         break;
      case TXF_FORMAT_BITMAP:
         width = txf->tex_width;
         height = txf->tex_height;
         stride = (width + 7) >> 3;
         texbitmap = (unsigned char *) malloc(stride * height);
         if (texbitmap == 0) {
            lastError = "out of memory.";
            goto error;
         }
         got = fread(texbitmap, 1, stride * height, file);
         EXPECT(stride * height);
         if (useLuminanceAlpha) {
            txf->teximage = (unsigned char *) calloc(width * height * 2, 1);
            if (txf->teximage == 0) {
               lastError = "out of memory.";
               goto error;
            }
            for (i = 0; i < height; i++) {
               for (j = 0; j < width; j++) {
                  if (texbitmap[i * stride + (j >> 3)] & (1 << (j & 7))) {
                     txf->teximage[(i * width + j) * 2] = 255;
                     txf->teximage[(i * width + j) * 2 + 1] = 255;
                  }
               }
            }
         } else {
            txf->teximage = (unsigned char *) calloc(width * height, 1);
            if (txf->teximage == 0) {
               lastError = "out of memory.";
               goto error;
            }
            for (i = 0; i < height; i++) {
               for (j = 0; j < width; j++) {
                  if (texbitmap[i * stride + (j >> 3)] & (1 << (j & 7))) {
                     txf->teximage[i * width + j] = 255;
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
      if (txf->tgi)
         free(txf->tgi);
      if (txf->tgvi)
         free(txf->tgvi);
      if (txf->lut)
         free(txf->lut);
      if (txf->teximage)
         free(txf->teximage);
      free(txf);
   }
   if (file)
      fclose(file);
   return 0;
}

/******************************************************************************/

//______________________________________________________________________________
GLuint txfEstablishTexture(TexFont * txf, GLuint texobj,
                           GLboolean setupMipmaps)
{
   if (txf->texobj == 0) {
      if (texobj == 0) {
         glGenTextures(1, &txf->texobj);
      } else {
         txf->texobj = texobj;
      }
   }
   glBindTexture(GL_TEXTURE_2D, txf->texobj);

   if (useLuminanceAlpha) {
      if (setupMipmaps) {
         gluBuild2DMipmaps(GL_TEXTURE_2D, GL_LUMINANCE_ALPHA,
                           txf->tex_width, txf->tex_height,
                           GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, txf->teximage);
      } else {
         glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE_ALPHA,
                      txf->tex_width, txf->tex_height, 0,
                      GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, txf->teximage);
      }
   } else {
      if (setupMipmaps) {
         gluBuild2DMipmaps(GL_TEXTURE_2D, GL_INTENSITY4,
                           txf->tex_width, txf->tex_height,
                           GL_LUMINANCE, GL_UNSIGNED_BYTE, txf->teximage);
      } else {
         glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY4,
                      txf->tex_width, txf->tex_height, 0,
                      GL_LUMINANCE, GL_UNSIGNED_BYTE, txf->teximage);
      }
   }

   // MT: tried changing MIN/MAG filters ... bad idea.

   return txf->texobj;
}

/******************************************************************************/

//______________________________________________________________________________
void txfBindFontTexture(TexFont * txf)
{
   glBindTexture(GL_TEXTURE_2D, txf->texobj);
}

/******************************************************************************/

//______________________________________________________________________________
void txfUnloadFont(TexFont * txf)
{
   if (txf->texobj) {
      glDeleteTextures(1, &txf->texobj);
   }
   if (txf->teximage) {
      free(txf->teximage);
   }
   free(txf->tgi);
   free(txf->tgvi);
   free(txf->lut);
   free(txf);
}

/******************************************************************************/

//______________________________________________________________________________
void txfGetStringMetrics(TexFont * txf, const char *TString, int len,
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
         w += int(tgvi->advance);
         ma = TMath::Max(ma, (int)( tgvi->v3[1]));
         md = TMath::Max(md, (int)(-tgvi->v0[1]));
      }
   }
   width = w;
   max_ascent  = ma; // txf->max_ascent;
   max_descent = md; // txf->max_descent;
   // printf("%d %d %d %d\n", txf->max_ascent, txf->max_descent, ma, md);
}

/******************************************************************************/

//______________________________________________________________________________
void txfRenderGlyph(TexFont * txf, int c)
{
   TexGlyphVertexInfo_t *tgvi;

   tgvi = getTCVI(txf, c);
   glBegin(GL_QUADS);
   glTexCoord2fv(tgvi->t0);
   glVertex2sv(tgvi->v0);
   glTexCoord2fv(tgvi->t1);
   glVertex2sv(tgvi->v1);
   glTexCoord2fv(tgvi->t2);
   glVertex2sv(tgvi->v2);
   glTexCoord2fv(tgvi->t3);
   glVertex2sv(tgvi->v3);
   glEnd();
   glTranslatef(tgvi->advance, 0.0, 0.0);
}

//______________________________________________________________________________
void txfRenderString(TexFont * txf, const char *TString, int len,
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
void txfRenderString(TexFont * txf, const char *TString, int len,
                     GLfloat maxx, GLfloat fadew,
                     bool keep_pos)
{
   GLfloat x = 0, xg0, xg1, yg0, yg1, f0, f1;
   fadew *= txf->max_width;
   GLfloat xfade = maxx - fadew;

   GLfloat col[4];
   glGetFloatv(GL_CURRENT_COLOR, col);

   glBegin(GL_QUADS);
   for (int i = 0; i < len; i++) {

      TexGlyphVertexInfo_t *tgvi;

      tgvi = getTCVI(txf, TString[i]);

      xg0 = x + tgvi->v0[0];
      xg1 = x + tgvi->v1[0];
      yg0 = tgvi->v0[1];
      yg1 = tgvi->v2[1];

      if(xg1 > xfade) {
         f0 = 1;        if(xg0 > xfade) f0 *= 1 - (xg0-xfade)/fadew;
         f1 = 1 - (xg1-xfade)/fadew;

         // printf("XX %s %c %f %f x(%f,%f) y(%f,%f)\n",
         //        TString, TString[i], f0, f1,
         //        xg0, xg1,yg0, yg1);

         glColor4f(f0*col[0], f0*col[1], f0*col[2], f0*col[3]);
         glTexCoord2fv(tgvi->t0);    glVertex2f(xg0, yg0);
         glColor4f(f1*col[0], f1*col[1], f1*col[2], f1*col[3]);
         glTexCoord2fv(tgvi->t1);    glVertex2f(xg1, yg0);
         glTexCoord2fv(tgvi->t2);    glVertex2f(xg1, yg1);
         glColor4f(f0*col[0], f0*col[1], f0*col[2], f0*col[3]);
         glTexCoord2fv(tgvi->t3);    glVertex2f(xg0, yg1);
      } else {
         glTexCoord2fv(tgvi->t0);    glVertex2f(xg0, yg0);
         glTexCoord2fv(tgvi->t1);    glVertex2f(xg1, yg0);
         glTexCoord2fv(tgvi->t2);    glVertex2f(xg1, yg1);
         glTexCoord2fv(tgvi->t3);    glVertex2f(xg0, yg1);
      }

      x += tgvi->advance;
      if(x > maxx) break;
   }
   glEnd();

   if(!keep_pos) glTranslatef(x, 0.0, 0.0);
}

/******************************************************************************/

//______________________________________________________________________________
void txfRenderGlyphZW(TexFont * txf, int c, float z, float w)
{
   TexGlyphVertexInfo_t *tgvi;

   tgvi = getTCVI(txf, c);
   glBegin(GL_QUADS);
   glTexCoord2fv(tgvi->t0);
   glVertex4f(tgvi->v0[0], tgvi->v0[1], z, w);
   glTexCoord2fv(tgvi->t1);
   glVertex4f(tgvi->v1[0], tgvi->v1[1], z, w);
   glTexCoord2fv(tgvi->t2);
   glVertex4f(tgvi->v2[0], tgvi->v2[1], z, w);
   glTexCoord2fv(tgvi->t3);
   glVertex4f(tgvi->v3[0], tgvi->v3[1], z, w);
   glEnd();
   glTranslatef(tgvi->advance, 0.0, 0.0);
}

//______________________________________________________________________________
void txfRenderStringZW(TexFont * txf, const char *TString, int len,
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
   MONO, TOP_BOTTOM, LEFT_RIGHT, FOUR
};

/******************************************************************************/

//______________________________________________________________________________
void txfRenderFancyString(TexFont * txf, char *TString, int len)
{
   TexGlyphVertexInfo_t *tgvi;
   GLubyte c[4][3];
   int mode = MONO;
   int i;

   for (i = 0; i < len; i++) {
      if (TString[i] == 27) {
         switch (TString[i + 1]) {
            case 'M':
               mode = MONO;
               glColor3ubv((GLubyte *) & TString[i + 2]);
               i += 4;
               break;
            case 'T':
               mode = TOP_BOTTOM;
               memcpy(c, &TString[i + 2], 6);
               i += 7;
               break;
            case 'L':
               mode = LEFT_RIGHT;
               memcpy(c, &TString[i + 2], 6);
               i += 7;
               break;
            case 'F':
               mode = FOUR;
               memcpy(c, &TString[i + 2], 12);
               i += 13;
               break;
         }
      } else {
         switch (mode) {
            case MONO:
               txfRenderGlyph(txf, TString[i]);
               break;
            case TOP_BOTTOM:
               tgvi = getTCVI(txf, TString[i]);
               glBegin(GL_QUADS);
               glColor3ubv(c[0]);
               glTexCoord2fv(tgvi->t0);
               glVertex2sv(tgvi->v0);
               glTexCoord2fv(tgvi->t1);
               glVertex2sv(tgvi->v1);
               glColor3ubv(c[1]);
               glTexCoord2fv(tgvi->t2);
               glVertex2sv(tgvi->v2);
               glTexCoord2fv(tgvi->t3);
               glVertex2sv(tgvi->v3);
               glEnd();
               glTranslatef(tgvi->advance, 0.0, 0.0);
               break;
            case LEFT_RIGHT:
               tgvi = getTCVI(txf, TString[i]);
               glBegin(GL_QUADS);
               glColor3ubv(c[0]);
               glTexCoord2fv(tgvi->t0);
               glVertex2sv(tgvi->v0);
               glColor3ubv(c[1]);
               glTexCoord2fv(tgvi->t1);
               glVertex2sv(tgvi->v1);
               glColor3ubv(c[1]);
               glTexCoord2fv(tgvi->t2);
               glVertex2sv(tgvi->v2);
               glColor3ubv(c[0]);
               glTexCoord2fv(tgvi->t3);
               glVertex2sv(tgvi->v3);
               glEnd();
               glTranslatef(tgvi->advance, 0.0, 0.0);
               break;
            case FOUR:
               tgvi = getTCVI(txf, TString[i]);
               glBegin(GL_QUADS);
               glColor3ubv(c[0]);
               glTexCoord2fv(tgvi->t0);
               glVertex2sv(tgvi->v0);
               glColor3ubv(c[1]);
               glTexCoord2fv(tgvi->t1);
               glVertex2sv(tgvi->v1);
               glColor3ubv(c[2]);
               glTexCoord2fv(tgvi->t2);
               glVertex2sv(tgvi->v2);
               glColor3ubv(c[3]);
               glTexCoord2fv(tgvi->t3);
               glVertex2sv(tgvi->v3);
               glEnd();
               glTranslatef(tgvi->advance, 0.0, 0.0);
               break;
         }
      }
   }
}

/******************************************************************************/

//______________________________________________________________________________
int txfInFont(TexFont * txf, int c)
{
   /* NOTE: No uppercase/lowercase substituion. */
   if ((c >= txf->min_glyph) && (c < txf->min_glyph + txf->range)) {
      if (txf->lut[c - txf->min_glyph]) {
         return 1;
      }
   }
   return 0;
}

/******************************************************************************/

//______________________________________________________________________________
bool LoadDefaultFont(const TString& file)
{
   static const TEveException _eh("TEveGLText::LoadFont ");

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
      throw(_eh + Form("Error loading font from file '%s': %s",
                       file.Data(), txfErrorString()));
   }

   return false;
}

} // end TEveGLText
