// @(#)root/gl:$Id$
// Author:  Olivier Couet  12/04/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RConfigure.h"
#include "TGLFontManager.h"

#include "TSystem.h"
#include "TEnv.h"
#include "TObjString.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"

// Direct inclussion of FTGL headers is deprecated in ftgl-2.1.3 while
// ftgl-2.1.2 shipped with root requires manual inclusion.
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

//______________________________________________________________________________
// TGLFont
//
// A wrapper class for FTFont.
// Holds pointer to FTFont object and its description: face size, font file
// and class ID. It  wraps Render and BBox functions.
//

ClassImp(TGLFont);

//______________________________________________________________________________
TGLFont::TGLFont():
   fFont(0), fManager(0), fDepth(0),
   fSize(0), fFile(0), fMode(kUndef),
   fTrashCount(0)
{
   // Constructor.
}

//______________________________________________________________________________
TGLFont::TGLFont(Int_t size, Int_t font, EMode mode, FTFont* f, TGLFontManager* mng):
   fFont(f), fManager(mng), fDepth(0),
   fSize(size), fFile(font), fMode(mode),
   fTrashCount(0)
{
   // Constructor.
}

//______________________________________________________________________________
TGLFont::TGLFont(const TGLFont &o):
   fFont(0), fManager(0), fDepth(0), fTrashCount(0)
{
   // Assignment operator.
   fFont = (FTFont*)o.GetFont();

   fSize  = o.fSize;
   fFile  = o.fFile;
   fMode  = o.fMode;

   fTrashCount = o.fTrashCount;
}

//______________________________________________________________________________
TGLFont::~TGLFont()
{
   //Destructor

   if (fManager) fManager->ReleaseFont(*this);
}

//______________________________________________________________________________
void TGLFont::CopyAttributes(const TGLFont &o)
{
   // Assignment operator.
   SetFont(o.fFont);
   SetManager(o.fManager);

   SetDepth(o.fDepth);

   fSize  = o.fSize;
   fFile  = o.fFile;
   fMode  = o.fMode;

   fTrashCount = o.fTrashCount;
}


/******************************************************************************/

//______________________________________________________________________________
void TGLFont::BBox(const char* txt, Float_t& llx, Float_t& lly, Float_t& llz, Float_t& urx, Float_t& ury, Float_t& urz) const
{
   // Get bounding box.

   // FTGL is not const correct.
   const_cast<FTFont*>(fFont)->BBox(txt, llx, lly, llz, urx, ury, urz);
}

//______________________________________________________________________________
void TGLFont::Render(const char* txt) const
{
   // Render text.

   Bool_t scaleDepth = (fMode == kExtrude && fDepth != 1.0f);

   if (scaleDepth) {
      glPushMatrix();
      // !!! 0.2*fSize is hard-coded in TGLFontManager::GetFont(), too.
      glTranslatef(0.0f, 0.0f, 0.5f*fDepth * 0.2f*fSize);
      glScalef(1.0f, 1.0f, fDepth);
   }

   // FTGL is not const correct.
   const_cast<FTFont*>(fFont)->Render(txt);

   if (scaleDepth) {
      glPopMatrix();
   }
}

//______________________________________________________________________________
void TGLFont::RenderBitmap(const char* txt, Float_t xs, Float_t ys, Float_t zs, ETextAlign_e align) const
{
   // Render text at the given position. Offset depends of text aligment.

   glPushMatrix();
   glTranslatef(xs, ys, zs);

   Float_t llx, lly, llz, urx, ury, urz;
   BBox(txt, llx, lly, llz, urx, ury, urz);
   if (txt[0] == '-')
      urx += (urx-llx)/strlen(txt);

   Float_t x=0, y=0;
   switch (align)
   {
      case kCenterUp:
         x = -urx*0.5; y = -ury;
         break;
      case kCenterDown:
         x = -urx*0.5; y = 0;
         break;
      case kRight:
         x = -urx; y =(lly -ury)*0.5;
         break;
      case kLeft:
         x = 0; y = -ury*0.5;
         break;
      default:
         break;
   };
   glRasterPos2i(0, 0);
   glBitmap(0, 0, 0, 0, x, y, 0);
   Render(txt);

   glPopMatrix();
}

//______________________________________________________________________________
void TGLFont::PreRender(Bool_t autoLight, Bool_t lightOn) const
{
   // Set-up GL state before FTFont rendering.

   switch (fMode)
   {
      case kBitmap:
      case kPixmap:
         glPushAttrib(GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT);
         glEnable(GL_ALPHA_TEST);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glAlphaFunc(GL_GEQUAL, 0.0625);
         break;
      case kTexture:
         glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
         glEnable(GL_TEXTURE_2D);
         glDisable(GL_CULL_FACE);
         glEnable(GL_ALPHA_TEST);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glAlphaFunc(GL_GEQUAL, 0.0625);
         break;
      case kOutline:
      case kPolygon:
      case kExtrude:
         glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
         glEnable(GL_NORMALIZE);
         glDisable(GL_CULL_FACE);
         break;
      default:
         Warning("TGLFont::PreRender", "Font mode undefined.");
         glPushAttrib(GL_LIGHTING_BIT);
         break;
   }

   if ((autoLight && fMode > TGLFont::kOutline) || (!autoLight && lightOn))
      glEnable(GL_LIGHTING);
   else
      glDisable(GL_LIGHTING);
}

//______________________________________________________________________________
void TGLFont::PostRender() const
{
   // Reset GL state after FTFont rendering.

   glPopAttrib();
}


//______________________________________________________________________________
//
// A FreeType GL font manager.
//
// Each GL rendering context has an instance of FTGLManager.
// This enables FTGL fonts to be shared same way as textures and display lists.

ClassImp(TGLFontManager);

TObjArray   TGLFontManager::fgFontFileArray;
TGLFontManager::FontSizeVec_t TGLFontManager::fgFontSizeArray;
Bool_t  TGLFontManager::fgStaticInitDone = kFALSE;

//______________________________________________________________________________
TGLFontManager::~TGLFontManager()
{
   // Destructor.

   FontMap_i it = fFontMap.begin();
   while (it != fFontMap.end()) {
      delete it->first.GetFont();
      it++;
   }
   fFontMap.clear();
}

//______________________________________________________________________________
void TGLFontManager::RegisterFont(Int_t size, Int_t fileID, TGLFont::EMode mode, TGLFont &out)
{
   // Provide font with given size, file and FTGL class.

   if (fgStaticInitDone == kFALSE) InitStatics();

   FontMap_i it = fFontMap.find(TGLFont(size, fileID, mode));
   if (it == fFontMap.end())
   {
      TString ttpath;
# ifdef TTFFONTDIR
      ttpath = gEnv->GetValue("Root.TTGLFontPath", TTFFONTDIR );
# else
      ttpath = gEnv->GetValue("Root.TTGLFontPath", "$(ROOTSYS)/fonts");
# endif
      TObjString* name = (TObjString*)fgFontFileArray[fileID];
      const char *file = gSystem->Which(ttpath.Data(), Form("%s.ttf", name->GetString().Data()));

      FTFont* ftfont = 0;
      switch (mode)
      {
         case TGLFont::kBitmap:
            ftfont = new FTGLBitmapFont(file);
            break;
         case TGLFont::kPixmap:
            ftfont = new FTGLPixmapFont(file);
            break;
         case TGLFont::kOutline:
            ftfont = new FTGLOutlineFont(file);
            break;
         case TGLFont::kPolygon:
            ftfont = new FTGLPolygonFont(file);
            break;
         case TGLFont::kExtrude:
            ftfont = new FTGLExtrdFont(file);
            ftfont->Depth(0.2*size);
            break;
         case TGLFont::kTexture:
            ftfont = new FTGLTextureFont(file);
            break;
         default:
            Error("TGLFontManager::GetFont", "invalid FTGL type");
            break;
      }
      delete [] file;
      ftfont->FaceSize(size);
      const TGLFont &mf = fFontMap.insert(std::make_pair(TGLFont(size, fileID, mode, ftfont, 0), 1)).first->first;
      out.CopyAttributes(mf);
   }
   else
   {
      if (it->first.GetTrashCount() > 0) {
         fFontTrash.remove(&(it->first));
         it->first.SetTrashCount(0);
      }
      ++(it->second);
      out.CopyAttributes(it->first);
   }
   out.SetManager(this);
}

//______________________________________________________________________________
void TGLFontManager::RegisterFont(Int_t size, const char* name, TGLFont::EMode mode, TGLFont &out)
{
   // Get mapping from ttf id to font names. Table taken from TTF.cxx.

   TObjArray* farr = GetFontFileArray();
   TIter next(farr);
   TObjString* os;
   Int_t cnt = 0;
   while ((os = (TObjString*) next()) != 0)
   {
      if (os->GetString() == name)
         break;
      cnt++;
   }

   if (cnt < farr->GetSize())
      RegisterFont(size, cnt, mode, out);
   else
      Error("TGLFontManager::GetFont", "unknown font name %s", name);
}

//______________________________________________________________________________
void TGLFontManager::ReleaseFont(TGLFont& font)
{
   // Release font with given attributes. Returns false if font has
   // not been found in the managers font set.

   FontMap_i it = fFontMap.find(font);

   if (it != fFontMap.end())
   {
      --(it->second);
      if (it->second == 0)
      {
         assert(it->first.GetTrashCount() == 0);
         it->first.IncTrashCount();
         fFontTrash.push_back(&it->first);
      }
   }
}

//______________________________________________________________________________
TObjArray* TGLFontManager::GetFontFileArray()
{
   // Get id to file name map.

   if (fgStaticInitDone == kFALSE) InitStatics();
   return &fgFontFileArray;
}

//______________________________________________________________________________
TGLFontManager::FontSizeVec_t* TGLFontManager::GetFontSizeArray()
{
   // Get valid font size vector.

   if (fgStaticInitDone == kFALSE) InitStatics();
   return &fgFontSizeArray;
}

//______________________________________________________________________________
Int_t TGLFontManager::GetFontSize(Float_t ds, Int_t min, Int_t max)
{
   // Get availabe font size.

   if (fgStaticInitDone == kFALSE) InitStatics();

   Int_t  nums = fgFontSizeArray.size();
   Int_t i = 0;
   while (i<nums)
   {
      if (ds<=fgFontSizeArray[i]) break;
      i++;
   }

   Int_t fs =  fgFontSizeArray[i];

   if (min>0 && fs<min)
      fs = min;

   if (max>0 && fs>max)
      fs = max;

   return fs;
}

//______________________________________________________________________________
const char* TGLFontManager::GetFontNameFromId(Int_t id)
{
   static const char *fonttable[] = {
      /* 0 */  "arialbd",
      /* 1 */  "timesi",
      /* 2 */  "timesbd",
      /* 3 */  "timesbi",
      /* 4 */  "arial",
      /* 5 */  "ariali",
      /* 6 */  "arialbd",
      /* 7 */  "arialbi",
      /* 8 */  "cour",
      /* 9 */  "couri",
      /*10 */  "courbd",
      /*11 */  "courbi",
      /*12 */  "symbol",
      /*13 */  "times",
      /*14 */  "wingding"
   };

   return fonttable[id / 10];
}

//______________________________________________________________________________
void TGLFontManager::InitStatics()
{
   // Create a list of available font files and allowed font sizes.

   const char *ttpath = gEnv->GetValue("Root.TTFontPath",
# ifdef TTFFONTDIR
                                       TTFFONTDIR);
# else
                                       "$(ROOTSYS)/fonts");
# endif

   void *dir = gSystem->OpenDirectory(ttpath);
   const char* name = 0;
   TString s;
   while ((name = gSystem->GetDirEntry(dir))) {
      s = name;
      if (s.EndsWith(".ttf")) {
         s.Resize(s.Sizeof() -5);
         fgFontFileArray.Add(new TObjString(s.Data()));
      }
   }
   fgFontFileArray.Sort();
   gSystem->FreeDirectory(dir);


   // font sizes
   for (Int_t i = 8; i <= 20; i+=2)
      fgFontSizeArray.push_back(i);
   for (Int_t i = 24; i <= 64; i+=4)
      fgFontSizeArray.push_back(i);
   for (Int_t i = 72; i <= 120; i+=8)
      fgFontSizeArray.push_back(i);

   fgStaticInitDone = kTRUE;
}

//______________________________________________________________________________
void TGLFontManager::ClearFontTrash()
{
   // Delete FTFFont objects registered for destruction.

   FontList_i it = fFontTrash.begin();
   while (it != fFontTrash.end())
   {
      if ((*it)->IncTrashCount() > 10000)
      {
         FontMap_i mi = fFontMap.find(**it);
         assert(mi != fFontMap.end());
         fFontMap.erase(mi);
         delete (*it)->GetFont();

         FontList_i li = it++;
         fFontTrash.erase(li);
      }
      else
      {
         ++it;
      }
   }
}
