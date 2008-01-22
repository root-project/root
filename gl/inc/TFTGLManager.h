#ifndef ROOT_TFTGLManager
#define ROOT_TFTGLManager

#include "TObjArray.h"
#include <set>

class FTFont;
class TRefCount;

class TFTGLManager
{
public:
   enum EMode { kBitmap, kPixmap, kOutline, kPolygon, kExtrude, kTexture }; // FTGL class

private:
   struct Font_t
   {
   public:
      Int_t      fSize;   // face size
      Int_t      fFile;   // file name
      EMode      fMode;   // FTGL class id

      FTFont*    fFont;   // FTGL font
      TRefCnt    fRefCnt; // FTGL font reference count

      Font_t(Int_t size, Int_t font,  EMode mode):
         fSize(size), fFile(font), fMode(mode), fFont(0), fRefCnt() {}

      Font_t(const Font_t& o):
         fSize(o.fSize), fFile(o.fFile), fMode(o.fMode), fFont(o.fFont), fRefCnt() { fRefCnt = o.fRefCnt; }

      Font_t& operator=(const Font_t& ref)
      {
         fSize   = ref.fSize;
         fFile   = ref.fFile;
         fMode   = ref.fMode;
         fFont   = ref.fFont;
         fRefCnt = ref.fRefCnt;
         return *this;
      }

      Bool_t operator<(const Font_t& o) const
      {
         if (fFile == o.fFile)
         {
            if(fMode == o.fMode)
            {
               return fSize < o.fSize;
            }
            return fMode < o.fMode;
         }
         return fFile < o.fFile;
      }
   }; // end inner struct Font_t

private:
   TFTGLManager(const TFTGLManager&);            // Not implemented
   TFTGLManager& operator=(const TFTGLManager&); // Not implemented

   std::set<Font_t>  fFontSet;          // Set of created fonts.

public:
   TFTGLManager(){}
   virtual ~TFTGLManager();

   FTFont*  GetFont(Int_t size, Int_t file, EMode mode);
   Bool_t   ReleaseFont(Int_t size, Int_t file, EMode mode);

   static Bool_t     fgStaticInitDone;  // Global initialization flag.
   static TObjArray  fgFontArray;       // Map font-id to ttf-font-file.
   static TObjArray* GetFontArray();
   static void       InitFontArray();

   ClassDef(TFTGLManager, 0); // A FreeType GL font manager.
};

#endif
