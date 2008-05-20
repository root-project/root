#ifndef ROOT_TGLFontManager
#define ROOT_TGLFontManager

#include "TObjArray.h"
#include <list>
#include <vector>
#include <map>

class FTFont;
class TGLFontManager;

class TGLFont
{
public:
   enum EMode
   {
      kUndef = -1,
      kBitmap, kPixmap,
      kTexture, kOutline, kPolygon, kExtrude
   }; // Font-types of FTGL.

private:
   TGLFont& operator=(const TGLFont& o); // Not implemented.

   FTFont          *fFont;     // FTGL font.
   TGLFontManager  *fManager;  // Font manager.

   Float_t          fDepth;  // depth of extruded fonts, enforced at render time.

protected:
   Int_t            fSize;   // free-type face size
   Int_t            fFile;   // free-type file name
   EMode            fMode;   // free-type FTGL class id

public:
   TGLFont();
   TGLFont(Int_t size, Int_t font, EMode mode, FTFont *f=0, TGLFontManager *mng=0);
   TGLFont(const TGLFont& o);            // Not implemented.
   virtual ~TGLFont();

   void CopyAttributes(const TGLFont &o);


   Int_t GetSize() const { return fSize;}
   Int_t GetFile() const { return fFile;}
   EMode GetMode() const { return fMode;}

   void  SetFont(FTFont *f) { fFont =f;}
   const FTFont* GetFont() const { return fFont; }
   void  SetManager(TGLFontManager *mng) {fManager = mng;}
   const TGLFontManager* GetManager() const { return fManager; }

   Float_t GetDepth() const { return fDepth;}
   void  SetDepth(Float_t d) { fDepth = d; }

   // FTGL wrapper functions
   void  BBox(const Text_t* txt,
               Float_t& llx, Float_t& lly, Float_t& llz,
               Float_t& urx, Float_t& ury, Float_t& urz) const;
   void  Render(const Text_t* txt) const;

   // helper gl draw functions
   virtual void PreRender(Bool_t autoLight=kTRUE, Bool_t lightOn=kFALSE) const;
   virtual void PostRender() const;

   Bool_t operator< (const TGLFont& o) const
   {
      if (fSize == o.fSize)
      {
         if(fFile == o.fFile)
         {
            return fMode < o.fMode;
         }
         return fFile < o.fFile;
      }
      return fSize < o.fSize;
   }

   ClassDef(TGLFont, 0); // A wrapper class for FTFont.
};

/******************************************************************************/
/******************************************************************************/

class TGLFontManager
{
public:
   typedef std::vector<Int_t> FontSizeVec_t;

private:
   TGLFontManager(const TGLFontManager&);            // Not implemented
   TGLFontManager& operator=(const TGLFontManager&); // Not implemented

   std::map<TGLFont, Int_t>  fFontMap;        // map of created fonts
   std::list<const FTFont*>  fFontTrash;      // fonts to purge

   static TObjArray     fgFontFileArray;      // map font-id to ttf-font-file
   static FontSizeVec_t fgFontSizeArray;      // map of valid font-size
   static Bool_t        fgStaticInitDone;     // global initialization flag
   static void          InitStatics();

public:
   TGLFontManager() : fFontMap(), fFontTrash() {}
   virtual ~TGLFontManager();

   void   RegisterFont(Int_t size, Int_t file, TGLFont::EMode mode, TGLFont& out);
   void   RegisterFont(Int_t size, const Text_t* name, TGLFont::EMode mode, TGLFont& out);
   void   ReleaseFont(TGLFont& font);

   static TObjArray*        GetFontFileArray();
   static FontSizeVec_t*    GetFontSizeArray();

   static Int_t             GetFontSize(Float_t ds, Int_t min = -1, Int_t max = -1);

   void   ClearFontTrash();

   ClassDef(TGLFontManager, 0); // A FreeType GL font manager.
};

#endif
