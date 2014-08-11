// @(#)root/graf:$Id$
// Author: Fons Rademakers, Reiner Rohlfs   15/10/2001

/*************************************************************************
 * Copyright (C) 2001-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TImage
#define ROOT_TImage


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TImage                                                               //
//                                                                      //
// Abstract interface to image processing library.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TAttImage
#include "TAttImage.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif
#ifndef ROOT_TVectorDfwd
#include "TVectorDfwd.h"
#endif

class TVirtualPad;
class TArrayD;
class TArrayL;
class TH2D;
class TPoint;
class TText;

class TImage : public TNamed, public TAttImage {

friend TImage operator+(const TImage &i1, const TImage &s2);
friend TImage operator/(const TImage &i1, const TImage &s2);

public:
   // Defines image file types
   enum EImageFileTypes {
      kXpm = 0,
      kZCompressedXpm,
      kGZCompressedXpm,
      kPng,
      kJpeg,
      kXcf,
      kPpm,
      kPnm,
      kBmp,
      kIco,
      kCur,
      kGif,
      kTiff,
      kXbm,
      kFits,
      kTga,
      kXml,
      kUnknown,
      kAnimGif
   };

   enum EText3DType {
      kPlain = 0,  // regular 2D text
      kEmbossed,
      kSunken,
      kShadeAbove,
      kShadeBelow,
      kEmbossedThick,
      kSunkenThick,
      kOutlineAbove,
      kOutlineBelow,
      kOutlineFull,
      k3DTypes
   };

   enum ECharType {
      kUTF8 = 0,
      kChar = 1,
      kUnicode = 4
   };

   enum ETileType {
      kStretch = 0,
      kTile,
      kStretchY,
      kStretchX
   };

   enum ECoordMode {
      kCoordModeOrigin = 0,
      kCoordModePrevious
   };

   enum EColorChan {
      kRedChan   = BIT(0),
      kGreenChan = BIT(1),
      kBlueChan  = BIT(2),
      kAlphaChan = BIT(3),
      kAllChan   = kRedChan | kGreenChan | kBlueChan | kAlphaChan
   };

protected:
   TImage(const char *file) : TNamed(file, "") { }
   TImage() { }

public:
   TImage(const TImage &img) : TNamed(img), TAttImage(img) { }
   TImage &operator=(const TImage &img)
            { TNamed::operator=(img); TAttImage::operator=(img); return *this; }
   TImage(UInt_t /*w*/, UInt_t /*h*/) : TNamed(), TAttImage() { }

   virtual ~TImage() { }

   // Cloning
   virtual TObject *Clone(const char *) const { return 0; }

   // Input / output
   virtual void ReadImage(const char * /*file*/, EImageFileTypes /*type*/ = TImage::kUnknown) {}
   virtual void WriteImage(const char * /*file*/, EImageFileTypes /*type*/ = TImage::kUnknown)  {}
   virtual void SetImage(const Double_t * /*imageData*/, UInt_t /*width*/, UInt_t /*height*/, TImagePalette * /*palette*/ = 0) {}
   virtual void SetImage(const TArrayD & /*imageData*/, UInt_t /*width*/, TImagePalette * /*palette*/ = 0) {}
   virtual void SetImage(const TVectorD & /*imageData*/, UInt_t /*width*/, TImagePalette * /*palette*/ = 0) {}
   virtual void SetImage(Pixmap_t /*pxm*/, Pixmap_t /*mask*/ = 0) {}

   // Create an image from the given pad. (See TASImage::FromPad)
   virtual void FromPad(TVirtualPad * /*pad*/, Int_t /*x*/ = 0, Int_t /*y*/ = 0, UInt_t /*w*/ = 0, UInt_t /*h*/ = 0) {}

   // Restore the image original size. (See TASImage::UnZoom)
   virtual void UnZoom() {}

   // Zoom the image. (See TASImage::Zoom)
   virtual void Zoom(UInt_t /*offX*/, UInt_t /*offY*/, UInt_t /*width*/, UInt_t /*height*/) {}

   // Flip the image by a multiple of 90 degrees. (See TASImage::Flip)
   virtual void Flip(Int_t /*flip*/ = 180) {}

   // Converts image to Gray. (See TASImage::Gray)
   virtual void Gray(Bool_t /*on*/ = kTRUE) {}
   virtual Bool_t IsGray() const { return kFALSE; }

   // Mirror the image. (See TASImage::Mirror)
   virtual void Mirror(Bool_t /*vert*/ = kTRUE) {}

   // Scale the image. (See TASImage::Scale)
   virtual void Scale(UInt_t /*width*/, UInt_t /*height*/) {}

   // Slice the image. (See TASImage::Slice)
   virtual void Slice(UInt_t /*xStart*/, UInt_t /*xEnd*/, UInt_t /*yStart*/,  UInt_t /*yEnd*/,
                      UInt_t /*toWidth*/, UInt_t /*toHeight*/) {}

   // Tile the image. (See TASImage::Tile)
   virtual void Tile(UInt_t /*width*/, UInt_t /*height*/) {}

   // Crop the image. (See TASImage::Crop)
   virtual void Crop(Int_t /*x*/ = 0, Int_t /*y*/ = 0, UInt_t /*width*/ = 0, UInt_t /*height*/ = 0) {}

   // Enlarge image. (See TASImage::Pad)
   virtual void Pad(const char * /*color*/ = "#FFFFFFFF", UInt_t /*left*/ = 0,
                   UInt_t /*right*/ = 0, UInt_t /*top*/ = 0, UInt_t /*bottom*/ = 0) {}

   // Gaussian blurr. (See TASImage::Blur)
   virtual void Blur(Double_t /*horizontal*/ = 3, Double_t /*vertical*/ = 3) { }

   // Reduces colordepth of an image. (See TASImage::Vectorize)
   virtual Double_t *Vectorize(UInt_t /*max_colors*/ = 256, UInt_t /*dither*/ = 4, Int_t /*opaque_threshold*/ = 0) { return 0; }

   // (See TASImage::HSV)
   virtual void HSV(UInt_t /*hue*/ = 0, UInt_t /*radius*/ = 360, Int_t /*H*/ = 0, Int_t /*S*/ = 0, Int_t /*V*/ = 0,
                    Int_t /*x*/ = 0, Int_t /*y*/ = 0, UInt_t /*width*/ = 0, UInt_t /*height*/ = 0) {}

   // Render multipoint gradient inside a rectangle. (See TASImage::Gradient)
   virtual void Gradient(UInt_t /*angle*/ = 0, const char * /*colors*/ = "#FFFFFF #000000", const char * /*offsets*/ = 0,
                         Int_t /*x*/ = 0, Int_t /*y*/ = 0, UInt_t /*width*/ = 0, UInt_t /*height*/ = 0) {}

   // Merge two images. (See TASImage::Merge)
   virtual void Merge(const TImage * /*im*/, const char * /*op*/ = "alphablend", Int_t /*x*/ = 0, Int_t /*y*/ = 0) {}

   // Append image. (See TASImage::Append)
   virtual void Append(const TImage * /*im*/, const char * /*option*/ = "+", const char * /*color*/ = "#00000000") {}

   // Bevel effect. (See TASImage::Bevel)
   virtual void Bevel(Int_t /*x*/ = 0, Int_t /*y*/ = 0, UInt_t /*width*/ = 0, UInt_t /*height*/ = 0,
                      const char * /*hi*/ = "#ffdddddd", const char * /*lo*/ = "#ff555555",
                      UShort_t /*thick*/ = 1, Bool_t /*pressed*/ = kFALSE) {}

   virtual void BeginPaint(Bool_t /*fast*/ = kTRUE) {}
   virtual void EndPaint() {}
   virtual void DrawLine(UInt_t /*x1*/, UInt_t /*y1*/, UInt_t /*x2*/, UInt_t /*y2*/,
                         const char * /*col*/ = "#000000", UInt_t /*thick*/ = 1) {}
   virtual void DrawDashLine(UInt_t /*x1*/, UInt_t /*y1*/, UInt_t /*x2*/, UInt_t /*y2*/, UInt_t /*nDash*/,
                             const char * /*pDash*/, const char * /*col*/ = "#000000", UInt_t /*thick*/ = 1) {}
   virtual void DrawBox(Int_t /*x1*/, Int_t /*y1*/, Int_t /*x2*/, Int_t /*y2*/,
                         const char * /*col*/ = "#000000", UInt_t /*thick*/ = 1, Int_t /*mode*/ = 0) {}
   virtual void DrawRectangle(UInt_t /*x*/, UInt_t /*y*/, UInt_t /*w*/, UInt_t /*h*/,
                              const char * /*col*/ = "#000000", UInt_t /*thick*/ = 1) {}
   virtual void FillRectangle(const char * /*col*/ = 0, Int_t /*x*/ = 0, Int_t /*y*/ = 0,
                              UInt_t /*width*/ = 0, UInt_t /*height*/ = 0) {}
   virtual void DrawPolyLine(UInt_t /*nn*/, TPoint * /*xy*/, const char * /*col*/ = "#000000",
                             UInt_t /*thick*/ = 1, TImage::ECoordMode /*mode*/ = kCoordModeOrigin) {}
   virtual void PutPixel(Int_t /*x*/, Int_t /*y*/, const char * /*col*/ = "#000000") {}
   virtual void PolyPoint(UInt_t /*npt*/, TPoint * /*ppt*/, const char * /*col*/ = "#000000",
                          TImage::ECoordMode /*mode*/ = kCoordModeOrigin) {}
   virtual void DrawSegments(UInt_t /*nseg*/, Segment_t * /*seg*/, const char * /*col*/ = "#000000", UInt_t /*thick*/ = 1) {}
   virtual void DrawText(Int_t /*x*/ = 0, Int_t /*y*/ = 0, const char * /*text*/ = "", Int_t /*size*/ = 12,
                         const char * /*color*/ = 0, const char * /*font*/ = "fixed",
                         EText3DType /*type*/ = TImage::kPlain, const char * /*fore_file*/ = 0, Float_t /*angle*/ = 0) { }
   virtual void DrawText(TText * /*text*/, Int_t /*x*/ = 0, Int_t /*y*/ = 0) { }
   virtual void FillPolygon(UInt_t /*npt*/, TPoint * /*ppt*/, const char * /*col*/ = "#000000",
                           const char * /*stipple*/ = 0, UInt_t /*w*/ = 16, UInt_t /*h*/ = 16) {}
   virtual void FillPolygon(UInt_t /*npt*/, TPoint * /*ppt*/, TImage * /*tile*/) {}
   virtual void CropPolygon(UInt_t /*npt*/, TPoint * /*ppt*/) {}
   virtual void DrawFillArea(UInt_t /*npt*/, TPoint * /*ppt*/, const char * /*col*/ = "#000000",
                           const char * /*stipple*/ = 0, UInt_t /*w*/ = 16, UInt_t /*h*/ = 16) {}
   virtual void DrawFillArea(UInt_t /*npt*/, TPoint * /*ppt*/, TImage * /*tile*/) {}
   virtual void FillSpans(UInt_t /*npt*/, TPoint * /*ppt*/, UInt_t * /*widths*/,  const char * /*col*/ = "#000000",
                         const char * /*stipple*/ = 0, UInt_t /*w*/ = 16, UInt_t /*h*/ = 16) {}
   virtual void FillSpans(UInt_t /*npt*/, TPoint * /*ppt*/, UInt_t * /*widths*/, TImage * /*tile*/) {}
   virtual void CropSpans(UInt_t /*npt*/, TPoint * /*ppt*/, UInt_t * /*widths*/) {}
   virtual void CopyArea(TImage * /*dst*/, Int_t /*xsrc*/, Int_t /*ysrc*/, UInt_t /*w*/, UInt_t /*h*/,
                         Int_t /*xdst*/ = 0, Int_t /*ydst*/ = 0, Int_t /*gfunc*/ = 3, EColorChan /*chan*/ = kAllChan) {}
   virtual void DrawCellArray(Int_t /*x1*/, Int_t /*y1*/, Int_t /*x2*/, Int_t /*y2*/, Int_t /*nx*/, Int_t /*ny*/, UInt_t * /*ic*/) {}
   virtual void FloodFill(Int_t /*x*/, Int_t /*y*/, const char * /*col*/, const char * /*min_col*/, const char * /*max_col*/ = 0) {}
   virtual void DrawCubeBezier(Int_t /*x1*/, Int_t /*y1*/, Int_t /*x2*/, Int_t /*y2*/, Int_t /*x3*/, Int_t /*y3*/, const char * /*col*/ = "#000000", UInt_t /*thick*/ = 1) {}
   virtual void DrawStraightEllips(Int_t /*x*/, Int_t /*y*/, Int_t /*rx*/, Int_t /*ry*/, const char * /*col*/ = "#000000", Int_t /*thick*/ = 1) {}
   virtual void DrawCircle(Int_t /*x*/, Int_t /*y*/, Int_t /*r*/, const char * /*col*/ = "#000000", Int_t /*thick*/ = 1) {}
   virtual void DrawEllips(Int_t /*x*/, Int_t /*y*/, Int_t /*rx*/, Int_t /*ry*/, Int_t /*angle*/, const char * /*col*/ = "#000000", Int_t /*thick*/ = 1) {}
   virtual void DrawEllips2(Int_t /*x*/, Int_t /*y*/, Int_t /*rx*/, Int_t /*ry*/, Int_t /*angle*/, const char * /*col*/ = "#000000", Int_t /*thick*/ = 1) {}

   virtual void SetEditable(Bool_t /*on*/ = kTRUE) {}
   virtual Bool_t IsEditable() const { return kFALSE; }

   virtual UInt_t GetWidth() const { return 0; }
   virtual UInt_t GetHeight() const { return 0; }
   virtual Bool_t IsValid() const { return kTRUE; }
   virtual TImage *GetScaledImage() const { return 0; }

   virtual TArrayL  *GetPixels(Int_t /*x*/= 0, Int_t /*y*/= 0, UInt_t /*w*/ = 0, UInt_t /*h*/ = 0) { return 0; }
   virtual TArrayD  *GetArray(UInt_t /*w*/ = 0, UInt_t /*h*/ = 0, TImagePalette * = gWebImagePalette) { return 0; }
   virtual Pixmap_t  GetPixmap() { return 0; }
   virtual Pixmap_t  GetMask() { return 0; }
   virtual UInt_t   *GetArgbArray() { return 0; }
   virtual UInt_t   *GetRgbaArray() { return 0; }
   virtual Double_t *GetVecArray() { return 0; }
   virtual UInt_t   *GetScanline(UInt_t /*y*/) { return 0; }
   virtual void      GetImageBuffer(char ** /*buffer*/, int* /*size*/, EImageFileTypes /*type*/ = TImage::kPng) {}
   virtual Bool_t    SetImageBuffer(char ** /*buffer*/, EImageFileTypes /*type*/ = TImage::kPng) { return kFALSE; }
   virtual void      PaintImage(Drawable_t /*wid*/, Int_t /*x*/, Int_t /*y*/, Int_t /*xsrc*/ = 0, Int_t /*ysrc*/ = 0, UInt_t /*wsrc*/ = 0, UInt_t /*hsrc*/ = 0, Option_t * /*opt*/ = "") { }
   virtual void      FromWindow(Drawable_t /*wid*/, Int_t /*x*/ = 0, Int_t /*y*/ = 0, UInt_t /*w*/ = 0, UInt_t /*h*/ = 0) {}
   virtual void      FromGLBuffer(UChar_t* /*buf*/, UInt_t /*w*/, UInt_t /*h*/) {}
   static EImageFileTypes GetImageFileTypeFromFilename(const char* opt);

   static TImage *Create();
   static TImage *Open(const char *file, EImageFileTypes type = kUnknown);
   static TImage *Open(char **data);
   static TImage *Open(const char *name, const Double_t *imageData, UInt_t width, UInt_t height, TImagePalette *palette);
   static TImage *Open(const char *name, const TArrayD &imageData, UInt_t width, TImagePalette *palette = 0);
   static TImage *Open(const char *name, const TVectorD &imageData, UInt_t width, TImagePalette *palette = 0);

   TImage    &operator+=(const TImage &i) { Append(&i, "+"); return *this; }
   TImage    &operator/=(const TImage &i) { Append(&i, "/"); return *this; }

   ClassDef(TImage,1)  // Abstract image class
};


#endif
