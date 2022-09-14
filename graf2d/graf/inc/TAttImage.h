// @(#)root/graf:$Id$
// Author: Reiner Rohlfs   24/03/02

/*************************************************************************
 * Copyright (C) 2001-2002, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttImage
#define ROOT_TAttImage

#include "TObject.h"

class TAttImage;

class TPaletteEditor {

protected:
   TAttImage    *fAttImage;    // image attributes to be edited

public:
   TPaletteEditor(TAttImage *attImage, UInt_t w, UInt_t h);
   virtual ~TPaletteEditor() { }

   virtual void CloseWindow();

   ClassDef(TPaletteEditor, 0)  // Base class for palette editor
};

class TImagePalette : public TObject {

public:
   UInt_t      fNumPoints;   ///< number of anchor points
   Double_t   *fPoints;      ///< [fNumPoints] value of each anchor point [0..1]
   UShort_t   *fColorRed;    ///< [fNumPoints] red color at each anchor point
   UShort_t   *fColorGreen;  ///< [fNumPoints] green color at each anchor point
   UShort_t   *fColorBlue;   ///< [fNumPoints] blue color at each anchor point
   UShort_t   *fColorAlpha;  ///< [fNumPoints] alpha at each anchor point

   TImagePalette();
   TImagePalette(const TImagePalette &palette);
   TImagePalette(UInt_t numPoints);
   TImagePalette(Int_t ncolors, Int_t *colors);
   virtual ~TImagePalette();
   virtual Int_t FindColor(UShort_t r, UShort_t g, UShort_t b);
   virtual Int_t *GetRootColors();

   TImagePalette &operator=(const TImagePalette &palette);

   static TImagePalette* Create(Option_t* opts);
   static TImagePalette* CreateCOLPalette(Int_t nContours);

   ClassDefOverride(TImagePalette,2)  // Color Palette for value -> color conversion
};

class TAttImage {

public:
   // Defines level of output quality/speed ratio
   enum EImageQuality {
      kImgDefault = -1,
      kImgPoor    = 0,
      kImgFast    = 1,
      kImgGood    = 2,
      kImgBest    = 3
   };

protected:
   EImageQuality    fImageQuality;       ///< *OPTION={GetMethod="GetImageQuality";SetMethod="SetImageQuality";Items=(kImgDefault="Default",kImgPoor="Poor",kImgFast="Fast",kImgGood="Good",kImgBest="Best")}*
   UInt_t           fImageCompression;   ///< compression [0 .. 100] 0: no compression
   Bool_t           fConstRatio;         ///< keep aspect ratio of image on the screen
   TImagePalette    fPalette;            ///< color palette for value -> color conversion
   TPaletteEditor  *fPaletteEditor;      ///<! GUI to edit the color palette
   Bool_t           fPaletteEnabled;     ///<! kTRUE - palette is drawn on the image

public:
   TAttImage();
   TAttImage(EImageQuality lquality, UInt_t lcompression, Bool_t constRatio);
   virtual ~TAttImage();

   void             Copy(TAttImage &attline) const;
   Bool_t           GetConstRatio() const { return fConstRatio; }
   UInt_t           GetImageCompression() const { return fImageCompression; }
   EImageQuality    GetImageQuality() const { return fImageQuality; }
   virtual const TImagePalette &GetPalette() const { return fPalette; }

   virtual void     ResetAttImage(Option_t *option="");
   virtual void     SaveImageAttributes(std::ostream &out, const char *name,
                                        EImageQuality qualdef = kImgDefault,
                                        UInt_t comprdef = 0,
                                        Bool_t constRatiodef = kTRUE);
   virtual void     SetConstRatio(Bool_t constRatio = kTRUE); // *TOGGLE*
   virtual void     SetPaletteEnabled(Bool_t on = kTRUE) {  fPaletteEnabled = on; }
   virtual void     SetImageCompression(UInt_t lcompression)
                     { fImageCompression = (lcompression > 100) ? 100 : lcompression; } // *MENU*
   virtual void     SetImageQuality(EImageQuality lquality)
                     { fImageQuality = lquality;} // *SUBMENU*
   virtual void     SetPalette(const TImagePalette *palette);
   virtual void     StartPaletteEditor(); // *MENU*
   virtual void     EditorClosed() { fPaletteEditor = nullptr; }
   Bool_t           IsPaletteEnabled() const { return fPaletteEnabled; }

   ClassDef(TAttImage,1)  //Image attributes
};

R__EXTERN TImagePalette  *gHistImagePalette;    // palette used in TH2::Draw("col")
R__EXTERN TImagePalette  *gWebImagePalette;     // 6x6x6 colors web palette

#endif
