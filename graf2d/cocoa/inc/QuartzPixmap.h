// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   16/02/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_QuartzPixmap
#define ROOT_QuartzPixmap

#include <vector>

#include <Cocoa/Cocoa.h>

#include "CocoaGuiTypes.h"
#include "CocoaUtils.h"
#include "X11Drawable.h"
#include "GuiTypes.h"

///////////////////////////////////////////////////////
//                                                   //
// "Pixmap". Graphical context to draw into image.   //
//                                                   //
///////////////////////////////////////////////////////

@interface QuartzPixmap : NSObject<X11Drawable> {
@private
   //32-bit Obj-C requires i-var to be declared (for a synthesized prop.).
   unsigned       fID;
   //

   unsigned       fWidth;
   unsigned       fHeight;

   std::vector<unsigned char> fData;
   ROOT::MacOSX::Util::CFScopeGuard<CGContextRef> fContext;

   CGFloat       fScaleFactor;
}

- (id) initWithW : (unsigned) width H : (unsigned) height scaleFactor : (CGFloat) scaleFactor;
- (BOOL) resizeW : (unsigned) width H : (unsigned) height scaleFactor : (CGFloat) scaleFactor;

- (CGImageRef) createImageFromPixmap;
- (CGImageRef) createImageFromPixmap : (ROOT::MacOSX::X11::Rectangle) cropArea;

//X11Drawable protocol.

@property (nonatomic, assign) unsigned fID;

- (BOOL) fIsPixmap;
- (BOOL) fIsOpenGLWidget;
- (CGFloat) fScaleFactor;

@property (nonatomic, readonly) CGContextRef fContext;

- (unsigned) fWidth;
- (unsigned) fHeight;

- (void) copy : (NSObject<X11Drawable> *) src area : (ROOT::MacOSX::X11::Rectangle) area withMask : (QuartzImage *) mask
         clipOrigin : (ROOT::MacOSX::X11::Point) origin toPoint : (ROOT::MacOSX::X11::Point) dstPoint;

- (unsigned char *) readColorBits : (ROOT::MacOSX::X11::Rectangle) area;

//
- (unsigned char *) fData;

//XPutPixel.
- (void) putPixel : (const unsigned char *) data X : (unsigned) x Y : (unsigned) y;
//XAddPixel.
- (void) addPixel : (const unsigned char *) rgb;

@end

/////////////////////////////////////////////////////////
//                                                     //
// CGImageRef, created from external data source (raw  //
// data)                                               //
//                                                     //
/////////////////////////////////////////////////////////

@interface QuartzImage : NSObject<X11Drawable> {
@private
   //32-bit Obj-C requires i-var to be declared (for a synthesized prop.).
   BOOL fIsStippleMask;
   unsigned fID;
   //
   unsigned       fWidth;
   unsigned       fHeight;

   ROOT::MacOSX::Util::CFScopeGuard<CGImageRef> fImage;
   std::vector<unsigned char> fImageData;
}

- (id) initWithW : (unsigned) width H : (unsigned) height data : (unsigned char *) data;
- (id) initMaskWithW : (unsigned) width H : (unsigned) height bitmapMask : (unsigned char *) mask;
- (id) initMaskWithW : (unsigned) width H : (unsigned) height;
- (id) initFromPixmap : (QuartzPixmap *) pixmap;
- (id) initFromImage : (QuartzImage *) image;
- (id) initFromImageFlipped : (QuartzImage *) image;

@property (nonatomic, readonly) BOOL fIsStippleMask;
- (CGImageRef) fImage;

//X11Drawable protocol.
@property (nonatomic, assign) unsigned fID;

- (BOOL) fIsPixmap;
- (BOOL) fIsOpenGLWidget;

- (unsigned) fWidth;
- (unsigned) fHeight;

- (unsigned char *) readColorBits : (ROOT::MacOSX::X11::Rectangle) area;

@end


namespace ROOT {
namespace MacOSX {
namespace X11 {

CGImageRef CreateSubImage(QuartzImage *image, const Rectangle &area);
//
bool AdjustCropArea(const Rectangle &srcRect, Rectangle &cropArea);
bool AdjustCropArea(QuartzImage *srcImage, Rectangle &cropArea);
bool AdjustCropArea(QuartzImage *srcImage, NSRect &cropArea);
bool AdjustCropArea(QuartzPixmap *srcImage, Rectangle &cropArea);

//Aux. function for TGCocoa.
void FillPixmapBuffer(const unsigned char *bitmap, unsigned width, unsigned height, ULong_t foregroundPixel,
                      ULong_t backgroundPixel, unsigned depth, unsigned char *imageData);

}
}
}

#endif
