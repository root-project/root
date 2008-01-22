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

TObjArray TFTGLManager::fgFontArray;
Bool_t    TFTGLManager::fgStaticInitDone = kFALSE;

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

   if (fgStaticInitDone == kFALSE) InitFontArray();

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
      TObjString* name = (TObjString*)fgFontArray[file];
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
TObjArray* TFTGLManager::GetFontArray()
{
   // Get id to file name map.

   if (fgStaticInitDone == kFALSE) InitFontArray();
   return &fgFontArray;
}

//______________________________________________________________________________
void TFTGLManager::InitFontArray()
{
   // Create a list of available font files.

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
         fgFontArray.Add(new TObjString(s.Data()));
      }
   }
   fgFontArray.Sort();
   gSystem->FreeDirectory(dir);

   fgStaticInitDone = kTRUE;
}
