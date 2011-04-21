// @(#)root/gl:$Id$
// Author: Alja Mrak-Tadel 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "RConfigure.h"
#include "TGLFontManager.h"


#include "TVirtualX.h"
#include "TMath.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TObjString.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"

// Direct inclussion of FTGL headers is deprecated in ftgl-2.1.3 while
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
Float_t TGLFont::GetAscent() const
{
   // Get font's ascent.

   return fFont->Ascender();
}

//______________________________________________________________________________
Float_t TGLFont::GetDescent() const
{
   // Get font's descent. The returned value is positive.

   return -fFont->Descender();
}

//______________________________________________________________________________
Float_t TGLFont::GetLineHeight() const
{
   // Get font's line-height.

   return fFont->LineHeight();
}

//______________________________________________________________________________
void TGLFont::MeasureBaseLineParams(Float_t& ascent, Float_t& descent, Float_t& line_height,
                                    const char* txt) const
{
   // Measure font's base-line parameters from the passed text.
   // Note that the measured parameters are not the same as the ones
   // returned by get-functions - those were set by the font designer.

   Float_t dum, lly, ury;
   const_cast<FTFont*>(fFont)->BBox(txt, dum, lly, dum, dum, ury, dum);
   ascent      =  ury;
   descent     = -lly;
   line_height =  ury - lly;
}

//______________________________________________________________________________
void TGLFont::BBox(const char* txt,
                   Float_t& llx, Float_t& lly, Float_t& llz,
                   Float_t& urx, Float_t& ury, Float_t& urz) const
{
   // Get bounding box.

   // FTGL is not const correct.
   const_cast<FTFont*>(fFont)->BBox(txt, llx, lly, llz, urx, ury, urz);
}

//______________________________________________________________________________
void TGLFont::Render(const char* txt, Double_t x, Double_t y, Double_t angle, Double_t /*mgn*/) const
{
   //mgn is simply ignored, because ROOT's TVirtualX TGX11 are complete mess with
   //painting attributes.
   glPushMatrix();
   //glLoadIdentity();
   
   // FTGL is not const correct.
   Float_t llx = 0.f, lly = 0.f, llz = 0.f, urx = 0.f, ury = 0.f, urz = 0.f;
   BBox(txt, llx, lly, llz, urx, ury, urz);
   
   /*
    V\H   | left | center | right
   _______________________________
   bottom |  7   |   8    |   9
   _______________________________
   center |  4   |   5    |   6
   _______________________________
    top   |  1   |   2    |   3
   */
   const Double_t dx = urx - llx, dy = ury - lly;
   Double_t xc = 0., yc = 0.;
   const UInt_t align = gVirtualX->GetTextAlign();

   switch (align) {
   case 7:
      xc += 0.5 * dx;
      yc += 0.5 * dy;
      break;
   case 8:
      yc += 0.5 * dy;
      break;
   case 9:
      xc -= 0.5 * dx;
      yc += 0.5 * dy;
      break;
   case 4:
      xc += 0.5 * dx;
      break;
   case 5:
      break;
   case 6:
      xc = -0.5 * dx;
      break;
   case 1:
      xc += 0.5 * dx;
      yc -= 0.5 * dy;
      break;
   case 2:
      yc -= 0.5 * dy;
      break;
   case 3:
      xc -= 0.5 * dx;
      yc -= 0.5 * dy;
      break;
   }
   
   glTranslated(x, y, 0.);
   glRotated(angle, 0., 0., 1.);
   glTranslated(xc, yc, 0.);
   glTranslated(-0.5 * dx, -0.5 * dy, 0.);
   //glScaled(mgn, mgn, 1.);
      
   const_cast<FTFont*>(fFont)->Render(txt);
   
   glPopMatrix();
   
   
}

//______________________________________________________________________________
void TGLFont::Render(const TString &txt) const
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
void  TGLFont:: Render(const TString &txt, Float_t x, Float_t y, Float_t z,
             ETextAlignH_e alignH, ETextAlignV_e alignV) const
{
   // Render text with given alignmentrepl and at given position.

   glPushMatrix();

   glTranslatef(x, y, z);

   x=0, y=0;
   Float_t llx, lly, llz, urx, ury, urz;
   BBox(txt, llx, lly, llz, urx, ury, urz);

   switch (alignH)
   {
      case TGLFont::kRight:
         x = -urx;
         break;

      case  TGLFont::kCenterH:
         x = -urx*0.5;
         break;
      default:
         break;
   };

   switch (alignV)
   {
      case TGLFont::kBottom:
         y = -ury;
         break;
      case  TGLFont::kCenterV:
         y = -ury*0.5;
         break;
      default:
         break;
   };

   if (fMode == TGLFont::kPixmap || fMode ==  TGLFont::kBitmap)
   {
      glRasterPos2i(0, 0);
      glBitmap(0, 0, 0, 0, x, y, 0);
   }
   else
   {
      glTranslatef(x, y, 0);
   }
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
void TGLFontManager::RegisterFont(Int_t sizeIn, Int_t fileID, TGLFont::EMode mode, TGLFont &out)
{
   // Provide font with given size, file and FTGL class.

   if (fgStaticInitDone == kFALSE) InitStatics();

   Int_t  size = GetFontSize(sizeIn);
   if (mode == out.GetMode() && fileID == out.GetFile() && size == out.GetSize())
      return;

   FontMap_i it = fFontMap.find(TGLFont(size, fileID, mode));
   if (it == fFontMap.end())
   {
      TString ttpath, file;
# ifdef TTFFONTDIR
      ttpath = gEnv->GetValue("Root.TTGLFontPath", TTFFONTDIR );
# else
      ttpath = gEnv->GetValue("Root.TTGLFontPath", "$(ROOTSYS)/fonts");
# endif
      {
         char *fp = gSystem->Which(ttpath, ((TObjString*)fgFontFileArray[fileID])->String() + ".ttf");
         file = fp;
         delete [] fp;
      }

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
            return;
            break;
      }
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
      if (os->String() == name)
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
Int_t TGLFontManager::GetFontSize(Int_t ds)
{
   // Get availabe font size.

   if (fgStaticInitDone == kFALSE) InitStatics();

   Int_t idx = TMath::BinarySearch(fgFontSizeArray.size(), &fgFontSizeArray[0],
                                   TMath::CeilNint(ds));
   if (idx < 0) idx = 0;
   return fgFontSizeArray[idx];
}

//______________________________________________________________________________
Int_t TGLFontManager::GetFontSize(Int_t ds, Int_t min, Int_t max)
{
   // Get availabe font size.

   if (ds < min) ds = min;
   if (ds > max) ds = max;
   return GetFontSize(ds);
}

//______________________________________________________________________________
const char* TGLFontManager::GetFontNameFromId(Int_t id)
{
   // Get font name from TAttAxis font id.

   if (fgStaticInitDone == kFALSE) InitStatics();

   TObjString* os = (TObjString*)fgFontFileArray[id / 10];
   return os->String().Data();
}

//______________________________________________________________________________
void TGLFontManager::InitStatics()
{
   // Create a list of available font files and allowed font sizes.

   fgFontFileArray.Add(new TObjString("arialbd"));  //   0

   fgFontFileArray.Add(new TObjString("timesi"));   //  10
   fgFontFileArray.Add(new TObjString("timesbd"));  //  20
   fgFontFileArray.Add(new TObjString("timesbi"));  //  30

   fgFontFileArray.Add(new TObjString("arial"));    //  40
   fgFontFileArray.Add(new TObjString("ariali"));   //  50
   fgFontFileArray.Add(new TObjString("arialbd"));  //  60
   fgFontFileArray.Add(new TObjString("arialbi"));  //  70

   fgFontFileArray.Add(new TObjString("cour"));     //  80
   fgFontFileArray.Add(new TObjString("couri"));    //  90
   fgFontFileArray.Add(new TObjString("courbd"));   // 100
   fgFontFileArray.Add(new TObjString("courbi"));   // 110

   fgFontFileArray.Add(new TObjString("symbol"));   // 120
   fgFontFileArray.Add(new TObjString("times"));    // 130
   fgFontFileArray.Add(new TObjString("wingding")); // 140


   // font sizes
   for (Int_t i = 8; i <= 20; i+=2)
      fgFontSizeArray.push_back(i);
   for (Int_t i = 24; i <= 64; i+=4)
      fgFontSizeArray.push_back(i);
   for (Int_t i = 72; i <= 128; i+=8)
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
