#include "TFTGLManager.h"

#include "TSystem.h"
#include "TEnv.h"
#include "TObjString.h"
#include "TGLUtil.h"
#include "TRefCnt.h"

#include "FTFont.h"
#include "FTGLExtrdFont.h"
#include "FTGLOutlineFont.h"
#include "FTGLPolygonFont.h"
#include "FTGLTextureFont.h"
#include "FTGLPixmapFont.h"
#include "FTGLBitmapFont.h"

//______________________________________________________________________________
// TFTGLManager
//
// A FreeType GL font manager.
//
// Each GL rendering context has an instance of FTGLManager.
// This enables FTGL fonts to be shared same way as textures and display lists.

ClassImp(TFTGLManager)

TObjArray   TFTGLManager::fgFontFileArray;
TFTGLManager::FontSizeVec_t TFTGLManager::fgFontSizeArray;
Bool_t  TFTGLManager::fgStaticInitDone = kFALSE;

TFTGLManager::~TFTGLManager()
{
   // Destructor.

   std::set<Font_t>::iterator it = fFontSet.begin();
   while (it != fFontSet.end()) {
      Font_t k = *it;
      if (k.fRefCnt.References() == 0)
      {
         delete k.fFont;
      }
      it++;
   }
}

//______________________________________________________________________________
FTFont* TFTGLManager::GetFont(Int_t size, Int_t file, EMode mode)
{
   // Provide font with given size, file and FTGL class.

   if (fgStaticInitDone == kFALSE) InitStatics();

   Font_t key = Font_t(size, file, mode);
   std::set<Font_t>::iterator it = fFontSet.find(key);
   if (it == fFontSet.end())
   {
      TString ttpath;
# ifdef TTFFONTDIR
      ttpath = gEnv->GetValue("Root.TTFontPath", TTFFONTDIR );
# else
      ttpath = gEnv->GetValue("Root.TTFontPath", "$(ROOTSYS)/fonts");
# endif
      TObjString* name = (TObjString*)fgFontFileArray[file];
      const char *file = gSystem->Which(ttpath.Data(), Form("%s.ttf", name->GetString().Data()));

      FTFont* ftfont = 0;
      switch (mode)
      {
         case kBitmap:
            ftfont = new FTGLBitmapFont(file);
            break;
         case kPixmap:
            ftfont = new FTGLPixmapFont(file);
            break;
         case kOutline:
            ftfont = new FTGLOutlineFont(file);
            break;
         case kPolygon:
            ftfont = new FTGLPolygonFont(file);
            break;
         case kExtrude:
            ftfont = new FTGLExtrdFont(file);
            ftfont->Depth(0.2*size);
            break;
         case kTexture:
            ftfont = new FTGLTextureFont(file);
            break;
         default:
            Error("TFTGLManager::GetFont", "invalid FTGL type");
            break;
      }
      ftfont->FaceSize(size);
      key.fFont = ftfont;
      fFontSet.insert(key);
   }
   else
   {
      key = *it;
   }
   key.fRefCnt.AddReference();
   return  key.fFont;
}

//______________________________________________________________________________
Bool_t TFTGLManager::ReleaseFont(Int_t size, Int_t file, EMode mode)
{
   // Release font with given attributes. Returns false if font has not been found
   // in the managers font set.

   std::set<Font_t>::iterator it = fFontSet.find(Font_t(size, file, mode));
   if(it != fFontSet.end())
   {
      Font_t k = *it;
      k.fRefCnt.RemoveReference();
      if (k.fRefCnt.References() == 0)
      {
         delete k.fFont;
         fFontSet.erase(k);
      }
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
TObjArray* TFTGLManager::GetFontFileArray()
{
   // Get id to file name map.

   if (fgStaticInitDone == kFALSE) InitStatics();
   return &fgFontFileArray;
}

//______________________________________________________________________________
TFTGLManager::FontSizeVec_t* TFTGLManager::GetFontSizeArray()
{
   // Get valid font size vector.

   if (fgStaticInitDone == kFALSE) InitStatics();
   return &fgFontSizeArray;
}

//______________________________________________________________________________
void TFTGLManager::InitStatics()
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

   fgStaticInitDone = kTRUE;
}

//______________________________________________________________________________
void TFTGLManager::PreRender(Int_t mode)
{
   // Set-up GL state before FTFont rendering.

   switch(mode) {
      case kBitmap:
      case kPixmap:
         glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT);
         glEnable(GL_ALPHA_TEST);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glAlphaFunc(GL_GEQUAL, 0.0625);
         break;
      case kTexture:
         glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
         glEnable(GL_TEXTURE_2D);
         glEnable(GL_COLOR_MATERIAL);
         glDisable(GL_CULL_FACE);
         glEnable(GL_ALPHA_TEST);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glAlphaFunc(GL_GEQUAL, 0.0625);
         break;
      case kExtrude:
      case kPolygon:
      case kOutline:
         glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
         glEnable(GL_NORMALIZE);
         glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
         glEnable(GL_COLOR_MATERIAL);
         glDisable(GL_CULL_FACE);
         break;
      default:
         break;
   }
}

//______________________________________________________________________________
void TFTGLManager::PostRender(Int_t mode)
{
   // Set-up GL state after FTFont rendering.

   switch(mode) {
      case kBitmap:
      case kPixmap:
         glPopAttrib();
         break;
      case kTexture:
         glPopAttrib();
         break;
      case kExtrude:
      case kPolygon:
      case kOutline:
         glPopAttrib();
         break;
      default:
         break;
      }
}
