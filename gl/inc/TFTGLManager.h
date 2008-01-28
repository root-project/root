#ifndef ROOT_TFTGLManager
#define ROOT_TFTGLManager

#include "TObjArray.h"
#include <set>
#include <vector>

class FTFont;
class TRefCount;

class TFTGLManager
{
public:
   enum EMode { kBitmap, kPixmap, kTexture, kOutline, kPolygon, kExtrude}; // FTGL class

   typedef std::vector<Int_t> FontSizeVec_t;

   struct Font_t
   {
      Int_t      fSize;   // face size
      Int_t      fFile;   // file name
      EMode      fMode;   // FTGL class id

      FTFont*    fFont;   // FTGL font
      TRefCnt    fRefCnt; // FTGL font reference count



      Font_t(): fSize(0), fFile(0), fMode(kPixmap), fFont(0), fRefCnt() {}

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

   static TObjArray     fgFontFileArray;       // Map font-id to ttf-font-file.
   static FontSizeVec_t fgFontSizeArray;
   static Bool_t        fgStaticInitDone;  // Global initialization flag.
   static void          InitStatics();

public:
   TFTGLManager(){}
   virtual ~TFTGLManager();

   FTFont*  GetFont(Int_t size, Int_t file, EMode mode);
   Bool_t   ReleaseFont(Int_t size, Int_t file, EMode mode);


   static TObjArray*        GetFontFileArray();
   static FontSizeVec_t*    GetFontSizeArray();
   static void              PreRender(Int_t mode);
   static void              PostRender(Int_t mode);

   ClassDef(TFTGLManager, 0); // A FreeType GL font manager.
};

#endif
