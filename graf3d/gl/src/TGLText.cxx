// @(#)root/gl:$Id$
// Author:  Olivier Couet  12/04/2007

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TError.h"

#include "TGLText.h"
#include "TGLUtil.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TGLIncludes.h"

// Direct inclusion of FTGL headers is deprecated in ftgl-2.1.3 while
// ftgl-2.1.2 shipped with ROOT requires manual inclusion.
#ifndef BUILTIN_FTGL
# include <FTGL/ftgl.h>
#else
# include "FTFont.h"
# include "FTGLExtrdFont.h"
# include "FTGLOutlineFont.h"
# include "FTGLPolygonFont.h"
# include "FTGLTextureFont.h"
# include "FTGLPixmapFont.h"
# include "FTGLBitmapFont.h"
#endif

#include <fontconfig/fontconfig.h>

#define FTGL_BITMAP  0
#define FTGL_PIXMAP  1
#define FTGL_OUTLINE 2
#define FTGL_POLYGON 3
#define FTGL_EXTRUDE 4
#define FTGL_TEXTURE 5

/** \class TGLText
\ingroup opengl
GL Text.

To draw a 3D text in a GL window. This class uses uses FTGL to render text.
FTGL is a package making the interface between the Free Type fonts and GL.
*/


////////////////////////////////////////////////////////////////////////////////

TGLText::TGLText()
{
   fX      = 0;
   fY      = 0;
   fZ      = 0;
   fAngle1 = 90;
   fAngle2 = 0;
   fAngle3 = 0;
   fGLTextFont = 0;
   SetGLTextFont(13); // Default font.
}


////////////////////////////////////////////////////////////////////////////////
/// TGLext normal constructor.

TGLText::TGLText(Double_t x, Double_t y, Double_t z, const char * /*text*/)
{
   fX      = x;
   fY      = y;
   fZ      = z;
   fAngle1 = 90;
   fAngle2 = 0;
   fAngle3 = 0;
   fGLTextFont = 0;
   SetGLTextFont(13); // Default font.
}


////////////////////////////////////////////////////////////////////////////////

TGLText::~TGLText()
{
   if (fGLTextFont) delete fGLTextFont;
}


////////////////////////////////////////////////////////////////////////////////
/// Draw text

void TGLText::PaintGLText(Double_t x, Double_t y, Double_t z, const char *text)
{
   if (!fGLTextFont) return;

   glPushMatrix();
   glTranslatef(x, y, z);

   TGLUtil::Color(GetTextColor());
   Double_t s = GetTextSize();
   glScalef(s,s,s);

   // Text alignment
   Float_t llx, lly, llz, urx, ury, urz;
   fGLTextFont->BBox(text, llx, lly, llz, urx, ury, urz);
   Short_t halign = fTextAlign/10;
   Short_t valign = fTextAlign - 10*halign;
   Float_t dx = 0, dy = 0;
   switch (halign) {
      case 1 : dx =  0    ; break;
      case 2 : dx = -urx/2; break;
      case 3 : dx = -urx  ; break;
   }
   switch (valign) {
      case 1 : dy =  0    ; break;
      case 2 : dy = -ury/2; break;
      case 3 : dy = -ury  ; break;
   }
   glTranslatef(dx, dy, 0);

   //In XY plane
   glRotatef(fAngle1,1.,0.,0.);

   //In XZ plane
   glRotatef(fAngle2,0.,1.,0.);

   //In YZ plane
   glRotatef(fAngle3,0.,0.,1.);

   // Render text
   fGLTextFont->Render(text);

   glPopMatrix();
}


////////////////////////////////////////////////////////////////////////////////

void TGLText::PaintBBox(const char *text)
{
   Float_t llx, lly, llz, urx, ury, urz;
   fGLTextFont->BBox(text, llx, lly, llz, urx, ury, urz);
   glBegin(GL_LINES);
   glVertex3f(   0,   0, 0); glVertex3f( urx,   0, 0);
   glVertex3f(   0,   0, 0); glVertex3f(   0, ury, 0);
   glVertex3f(   0, ury, 0); glVertex3f( urx, ury, 0);
   glVertex3f( urx, ury, 0); glVertex3f( urx,   0, 0);
   glEnd();
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate bounding-box for given string.

void TGLText::BBox(const char* string, float& llx, float& lly, float& llz,
                                       float& urx, float& ury, float& urz)
{
   fGLTextFont->BBox(string, llx, lly, llz, urx, ury, urz);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the text rotation angles.

void TGLText::SetGLTextAngles(Double_t a1, Double_t a2, Double_t a3)
{
   fAngle1 = a1;
   fAngle2 = a2;
   fAngle3 = a3;
}


////////////////////////////////////////////////////////////////////////////////

void TGLText::SetGLTextFont(Font_t fontnumber)
{
   int fontid = fontnumber / 10;

   static const char *fonttable[] = {
      "freesans:bold",
      "freeserif:italic",
      "freeserif:bold",
      "freeserif:bold:italic",
      "freesans",
      "freesans:italic",
      "freesans:bold",
      "freesans:bold:italic",
      "freemono",
      "freemono:italic",
      "freemono:bold",
      "freemono:bold:italic",
      "standardsymbolsps",
      "freeserif",
      "dingbats",
   };

   char *ttfont;

   FcPattern *pat, *match;
   FcResult result;

   pat = FcNameParse ((const FcChar8*) fonttable[fontid]);

   FcConfigSubstitute (nullptr, pat, FcMatchPattern);
   FcDefaultSubstitute (pat);
   match = FcFontMatch (nullptr, pat, &result);
   FcPatternGetString (match, FC_FILE, 0, (FcChar8**) &ttfont);

   if (fGLTextFont) delete fGLTextFont;

// fGLTextFont = new FTGLOutlineFont(ttfont);

   fGLTextFont = new FTGLPolygonFont(ttfont);

   FcPatternDestroy (match);
   FcPatternDestroy (pat);

   if (!fGLTextFont->FaceSize(1))
      Error("SetGLTextFont","Cannot set FTGL::FaceSize");
}
