// @(#)root/gui:$Id$
// Author: Fons Rademakers   05/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGIcon
#define ROOT_TGIcon


#include "TGFrame.h"
#include "TGDimension.h"

class TGPicture;
class TImage;

class TGIcon : public TGFrame {

protected:
   const TGPicture  *fPic;     ///< icon picture
   TImage           *fImage;   ///< image
   TString           fPath;    ///< directory of image

   void DoRedraw() override;

private:
   TGIcon(const TGIcon &) = delete;
   TGIcon& operator=(const TGIcon&) = delete;

public:
   TGIcon(const TGWindow *p, const TGPicture *pic, UInt_t w, UInt_t h,
      UInt_t options = kChildFrame, Pixel_t back = GetDefaultFrameBackground()) :
         TGFrame(p, w, h, options, back), fPic(pic), fImage(nullptr), fPath() { SetWindowName(); }

   TGIcon(const TGWindow *p = nullptr, const char *image = nullptr);

   virtual ~TGIcon();

   virtual void Reset();         //*MENU*
   const TGPicture *GetPicture() const { return fPic; }
   TImage *GetImage() const { return fImage; }
   virtual void SetPicture(const TGPicture *pic);
   virtual void SetImage(const char *img);
   virtual void SetImage(TImage *img);
   virtual void SetImagePath(const char *path);

   void Resize(UInt_t w = 0, UInt_t h = 0) override;
   void Resize(TGDimension size) override { Resize(size.fWidth, size.fHeight); }
   void MoveResize(Int_t x, Int_t y, UInt_t w = 0, UInt_t h = 0) override;
   virtual void ChangeBackgroundColor() {}

   TGDimension GetDefaultSize() const override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGIcon,0)  // Icon GUI class
};

#endif
