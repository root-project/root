// $Id: TGHtmlImage.cxx,v 1.2 2007/05/07 15:28:48 brun Exp $
// Author:  Valeriy Onuchin   03/05/2007

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**************************************************************************

    HTML widget for xclass. Based on tkhtml 1.28
    Copyright (C) 1997-2000 D. Richard Hipp <drh@acm.org>
    Copyright (C) 2002-2003 Hector Peraza.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

// Routines used for processing <IMG> markup

#include <string.h>
#include <stdlib.h>

#include "TGHtml.h"
//#include <TGHtmlUri.h>
#include "TImage.h"
#include "TUrl.h"
#include "TSocket.h"
#include "TSystem.h"

//______________________________________________________________________________
TGHtmlImage::TGHtmlImage(TGHtml *h, const char *url, const char *width,
                       const char *height)
{
   // ctor.

   _html = h;
   zUrl = StrDup(url);
   zWidth = StrDup(width);
   zHeight = StrDup(height);
   image = NULL;
   pNext = NULL;
   pList = NULL;
   w = 0;
   h = 0;
   timer = NULL;
}

//______________________________________________________________________________
TGHtmlImage::~TGHtmlImage()
{
   //  dtor.

   delete [] zUrl;
   delete [] zWidth;
   delete [] zHeight;

   if (image) delete image;
   if (timer) delete timer;
}

//______________________________________________________________________________
int TGHtml::GetImageAlignment(TGHtmlElement *p)
{
   // Find the alignment for an image

   char *z;
   int i;
   int result;

   static struct {
      char *zName;
      int iValue;
   } aligns[] = {
      { "bottom",    IMAGE_ALIGN_Bottom    },
      { "baseline",  IMAGE_ALIGN_Bottom    },
      { "middle",    IMAGE_ALIGN_Middle    },
      { "top",       IMAGE_ALIGN_Top       },
      { "absbottom", IMAGE_ALIGN_AbsBottom },
      { "absmiddle", IMAGE_ALIGN_AbsMiddle },
      { "texttop",   IMAGE_ALIGN_TextTop   },
      { "left",      IMAGE_ALIGN_Left      },
      { "right",     IMAGE_ALIGN_Right     },
   };

   z = p->MarkupArg("align", 0);
   result = IMAGE_ALIGN_Bottom;
   if (z) {
      for (i = 0; i < int(sizeof(aligns) / sizeof(aligns[0])); i++) {
         if (strcasecmp(aligns[i].zName, z) == 0) {
            result = aligns[i].iValue;
            break;
         }
      }
   }
   return result;
}

//______________________________________________________________________________
void TGHtml::ImageChanged(TGHtmlImage *pImage, int newWidth, int newHeight)
{
   // This routine is called when an image changes. If the size of the
   // images changes, then we need to completely redo the layout. If
   // only the appearance changes, then this works like an expose event.
   //
   // pImage    - Pointer to an TGHtmlImage object
   // newWidth  - New width of the image
   // newHeight - New height of the image

   TGHtmlImageMarkup *pElem;

   if (pImage->w != newWidth || pImage->h != newHeight) {
      // We have to completely redo the layout after adjusting the size
      // of the images
      for (pElem = pImage->pList; pElem; pElem = pElem->iNext) {
         pElem->w = newWidth;
         pElem->h = newHeight;
      }
      flags |= RELAYOUT;
      pImage->w = newWidth;
      pImage->h = newHeight;
      RedrawEverything();
   } else {
#if 0
    for (pElem = pImage->pList; pElem; pElem = pElem->iNext) {
      pElem->redrawNeeded = 1;
    }
    flags |= REDRAW_IMAGES;
    ScheduleRedraw();
#else
      for (pElem = pImage->pList; pElem; pElem = pElem->iNext) {
         pElem->redrawNeeded = 1;
         DrawRegion(pElem->x, pElem->y - pElem->ascent, pElem->w, pElem->h);
      }
#endif
   }
}

//______________________________________________________________________________
TGHtmlImage *TGHtml::GetImage(TGHtmlImageMarkup *p)
{
   // Given an <IMG> markup, find or create an appropriate TGHtmlImage
   // object and return a pointer to that object. NULL might be returned.

   char *zWidth;
   char *zHeight;
   char *zSrc;
   TGHtmlImage *pImage;

   if (p->type != Html_IMG) { CANT_HAPPEN; return 0; }

   zSrc = p->MarkupArg("src", 0);
   if (zSrc == 0) return 0;

   zSrc = ResolveUri(zSrc);
   if (zSrc == 0) return 0;

   zWidth = p->MarkupArg("width", "");
   zHeight = p->MarkupArg("height", "");

   //p->w = atoi(zWidth);
   //p->h = atoi(zHeight);

   for (pImage = imageList; pImage; pImage = pImage->pNext) {
      if (strcmp(pImage->zUrl, zSrc) == 0
          &&  strcmp(pImage->zWidth, zWidth) == 0
          &&  strcmp(pImage->zHeight, zHeight) == 0) {
         delete [] zSrc;
         return pImage;
      }
   }

   TImage *img = LoadImage(zSrc, atoi(zWidth), atoi(zHeight));

   if (img) {
      pImage = new TGHtmlImage(this, zSrc, zWidth, zHeight);
      pImage->image = img;
    //if (img->IsAnimated()) {
    //  pImage->timer = new TTimer(this, img->GetAnimDelay());
    //}
      ImageChanged(pImage, img->GetWidth(), img->GetHeight());
      pImage->pNext = imageList;
      imageList = pImage;
   } else {
      pImage = 0;
   }

   delete [] zSrc;

   return pImage;
}

//______________________________________________________________________________
static TImage *ReadRemoteImage(const char *url)
{
   // Temporary function to read remote pictures

   TImage *image = 0;
   FILE *tmp;
   char *buf;
   TUrl fUrl(url);

   TString msg = "GET ";
   msg += fUrl.GetProtocol();
   msg += "://";
   msg += fUrl.GetHost();
   msg += ":";
   msg += fUrl.GetPort();
   msg += "/";
   msg += fUrl.GetFile();
   msg += "\r\n";

   TString uri(url);
   if (!uri.BeginsWith("http://") || uri.EndsWith(".html"))
      return 0;
   TSocket s(fUrl.GetHost(), fUrl.GetPort());
   if (!s.IsValid())
      return 0;
   if (s.SendRaw(msg.Data(), msg.Length()) == -1)
      return 0;
   Int_t size = 1024*1024;
   buf = (char *)calloc(size, sizeof(char));
   if (s.RecvRaw(buf, size) == -1) {
      free(buf);
      return 0;
   }
   TString pathtmp = Form("%s/%s",
      gSystem->TempDirectory(), gSystem->BaseName(url));
   tmp = fopen(pathtmp.Data(), "wb");
   if (!tmp) {
      free(buf);
      return 0;
   }
   fwrite(buf, sizeof(char), size, tmp);
   fclose(tmp);
   free(buf);
   image = TImage::Open(pathtmp.Data());
   if (!image->IsValid()) {
      delete image;
      image = 0;
   }
   gSystem->Unlink(pathtmp.Data());
   return image;
}

//______________________________________________________________________________
TImage *TGHtml::LoadImage(const char *url, int w, int h)
{
   // This is the default LoadImage() procedure. It just tries to load the
   // image from a file in the local filesystem.

   TImage *image = 0;

   //TGHtmlUri uri(url);

   TString uri(url);
   if (uri.BeginsWith("http://") && !uri.EndsWith(".html"))
      image = ReadRemoteImage(url);
   else
      image = TImage::Open(url);
   if (image) {
      if (!image->IsValid()) {
         delete image;
         image = 0;
         return 0;
      }
      if ((w > 0 && h > 0) && ((w != (int)image->GetWidth()) ||
          (h != (int)image->GetHeight()))) {
         image->Scale(w, h);
      }
   }
   return image;
}

//______________________________________________________________________________
char *TGHtml::GetPctWidth(TGHtmlElement *p, char *opt, char *ret)
{
   // Return the height and width, converting to percent if required

   int n, m, val;
   char *tz, *z;
   TGHtmlElement *pElem = p;

   z = pElem->MarkupArg(opt, "");
   if (!strchr(z, '%')) return z;
   if (!sscanf(z, "%d", &n)) return z;
   if (n < 0 || n > 100) return z;
   if (opt[0] == 'h') {
      val = fCanvas->GetHeight() * 100;
   } else {
      val = fCanvas->GetWidth() * 100;
   }
   if (!inTd) {
      sprintf(ret, "%d", val / n);
   } else {
      while (pElem && pElem->type != Html_TD) pElem = pElem->pPrev;
      if (!pElem) return z;
      tz = pElem->MarkupArg(opt, 0);
      if (tz && !strchr(tz, '%') && sscanf(tz, "%d", &m)) {
         sprintf(ret, "%d", m * 100 / n);
         return ret;
      }
      pElem = ((TGHtmlCell *)pElem)->pTable;
      if (!pElem) return z;
      tz = pElem->MarkupArg(opt, 0);
      if (tz && !strchr(tz, '%') && sscanf(tz, "%d", &m)) {
         sprintf(ret, "%d", m * 100 / n);
         return ret;
      }
      return z;
   }
   return ret;
}

//______________________________________________________________________________
int TGHtml::GetImageAt(int x, int y)
{
   // This routine searchs for an image beneath the coordinates x,y
   // and returns the token number of the the image, or -1 if no
   // image found.

   TGHtmlBlock *pBlock;
   TGHtmlElement *pElem;
   //int n;

   for (pBlock = firstBlock; pBlock; pBlock = pBlock->bNext) {
      if (pBlock->top > y || pBlock->bottom < y ||
          pBlock->left > x || pBlock->right < x) {
         continue;
      }
      for (pElem = pBlock->pNext; pElem; pElem = pElem->pNext) {
         if (pBlock->bNext && pElem == pBlock->bNext->pNext) break;
         if (pElem->type == Html_IMG) {
            return TokenNumber(pElem);
         }
      }
   }

   return -1;
}
