// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   16/02/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#define NDEBUG

#import <algorithm>
#import <utility>
#import <cassert>
#import <cstddef>
#import <limits>
#import <new>

#import "CocoaGuiTypes.h"
#import "QuartzWindow.h"
#import "QuartzPixmap.h"
#import "QuartzUtils.h"
#import "CocoaUtils.h"
#import "X11Colors.h"

//Call backs for data provider.
extern "C" {

//______________________________________________________________________________
const void* ROOT_QuartzImage_GetBytePointer(void *info)
{
   assert(info != 0 && "ROOT_QuartzImage_GetBytePointer, info parameter is null");
   return info;
}

//______________________________________________________________________________
void ROOT_QuartzImage_ReleaseBytePointer(void *, const void *)
{
   //Do nothing.
}

//______________________________________________________________________________
std::size_t ROOT_QuartzImage_GetBytesAtPosition(void* info, void* buffer, off_t position, std::size_t count)
{
    std::copy((char *)info + position, (char *)info + position + count, (char*)buffer);
    return count;
}

}

namespace X11 = ROOT::MacOSX::X11;
namespace Util = ROOT::MacOSX::Util;
namespace Quartz = ROOT::Quartz;

@implementation QuartzPixmap {
@private
   unsigned       fWidth;
   unsigned       fHeight;
   unsigned char *fData;
   CGContextRef   fContext;
   
   unsigned       fScaleFactor;
}

@synthesize fID;

//______________________________________________________________________________
- (id) initWithW : (unsigned) width H : (unsigned) height scaleFactor : (CGFloat) scaleFactor
{
   if (self = [super init]) {
      fWidth = 0;
      fHeight = 0;
      fData = 0;
      fContext = 0;
      
      if (![self resizeW : width H : height scaleFactor : scaleFactor]) {
         [self release];
         return nil;
      }
   }

   return self;
}

//______________________________________________________________________________
- (void) dealloc
{
   if (fContext)
      CGContextRelease(fContext);

   delete [] fData;

   [super dealloc];
}

//______________________________________________________________________________
- (BOOL) resizeW : (unsigned) width H : (unsigned) height scaleFactor : (CGFloat) scaleFactor
{
   assert(width > 0 && "resizeW:H:, Pixmap width must be positive");
   assert(height > 0 && "resizeW:H:, Pixmap height must be positive");

   fScaleFactor = unsigned(scaleFactor + 0.5);
   
   //Part, which does not change anything in a state:
   unsigned char *memory = 0;
   
   const unsigned scaledW = width * fScaleFactor;
   const unsigned scaledH = height * fScaleFactor;
   
   try {
      memory = new unsigned char[scaledW * scaledH * 4]();//[0]
   } catch (const std::bad_alloc &) {
      NSLog(@"QuartzPixmap: -resizeW:H:, memory allocation failed");
      return NO;
   }
   
   Util::ScopedArray<unsigned char> arrayGuard(memory);

   const Util::CFScopeGuard<CGColorSpaceRef> colorSpace(CGColorSpaceCreateDeviceRGB());//[1]
   if (!colorSpace.Get()) {
      NSLog(@"QuartzPixmap: -resizeW:H:, CGColorSpaceCreateDeviceRGB failed");
      return NO;
   }

   Util::CFScopeGuard<CGContextRef> ctx(CGBitmapContextCreateWithData(memory, scaledW, scaledH, 8, scaledW * 4, colorSpace.Get(), kCGImageAlphaPremultipliedLast, NULL, 0));
   if (!ctx.Get()) {
      NSLog(@"QuartzPixmap: -resizeW:H:, CGBitmapContextCreateWithData failed");
      return NO;
   }
   
   //Now, apply scaling.
   
   if (fScaleFactor > 1)
      CGContextScaleCTM(ctx.Get(), fScaleFactor, fScaleFactor);

   //All initializations are OK, now change the state:
   if (fContext) {
      //New context was created OK, we can release now the old one.
      CGContextRelease(fContext);//[2]
   }

   //Release old memory.
   delete [] fData;

   //sizes, data.
   fWidth = width;
   fHeight = height;
   fData = memory;
   
   arrayGuard.Release();

   fContext = ctx.Get();//[2]
   ctx.Release();//Stop the ownership.

   return YES;
}

//______________________________________________________________________________
- (CGImageRef) createImageFromPixmap
{
   return [self createImageFromPixmap : X11::Rectangle(0, 0, fWidth, fHeight)];
}

//______________________________________________________________________________
- (CGImageRef) createImageFromPixmap : (X11::Rectangle) cropArea
{
   //Crop area must be valid and adjusted by caller.
   
   //This function is incorrect in a general case, it does not care about
   //cropArea.fX and cropArea.fY, very sloppy implementation.
   //TODO: either fix it or remove completely.
   
   assert(cropArea.fX >= 0 && "createImageFromPixmap:, cropArea.fX is negative");
   assert(cropArea.fY >= 0 && "createImageFromPixmap:, cropArea.fY is negative");
   assert(cropArea.fWidth <= fWidth && "createImageFromPixmap:, bad cropArea.fWidth");
   assert(cropArea.fHeight <= fHeight && "createImageFromPixmap:, bad cropArea.fHeight");

   //
   const CGDataProviderDirectCallbacks providerCallbacks = {0, ROOT_QuartzImage_GetBytePointer, 
                                                            ROOT_QuartzImage_ReleaseBytePointer, 
                                                            ROOT_QuartzImage_GetBytesAtPosition, 0};

   const unsigned scaledW = fWidth * fScaleFactor;
   const unsigned scaledH = fHeight * fScaleFactor;
   
   
   const Util::CFScopeGuard<CGDataProviderRef> provider(CGDataProviderCreateDirect(fData, scaledW * scaledH * 4, &providerCallbacks));
   if (!provider.Get()) {
      NSLog(@"QuartzPixmap: -pixmapToImage, CGDataProviderCreateDirect failed");
      return 0;
   }

   //RGB - this is only for TGCocoa::CreatePixmapFromData.
   const Util::CFScopeGuard<CGColorSpaceRef> colorSpace(CGColorSpaceCreateDeviceRGB());
   if (!colorSpace.Get()) {
      NSLog(@"QuartzPixmap: -pixmapToImage, CGColorSpaceCreateDeviceRGB failed");
      return 0;
   }
      
   //8 bits per component, 32 bits per pixel, 4 bytes per pixel, kCGImageAlphaLast:
   //all values hardcoded for TGCocoa.
   CGImageRef image = CGImageCreate(cropArea.fWidth * fScaleFactor, cropArea.fHeight * fScaleFactor, 8, 32, fWidth * 4 * fScaleFactor, colorSpace.Get(), kCGImageAlphaPremultipliedLast, provider.Get(), 0, false, kCGRenderingIntentDefault);

   return image;
}

//______________________________________________________________________________
- (BOOL) fIsPixmap
{
   return YES;
}

//______________________________________________________________________________
- (BOOL) fIsOpenGLWidget
{
   return NO;
}

//______________________________________________________________________________
- (CGContextRef) fContext
{
   assert(fContext != 0 && "fContext, called for bad pixmap");

   return fContext;
}

//______________________________________________________________________________
- (unsigned) fWidth
{
   assert(fContext != 0 && "fWidth, called for bad pixmap");

   return fWidth;
}

//______________________________________________________________________________
- (unsigned) fHeight
{
   assert(fContext != 0 && "fHeight, called for bad pixmap");

   return fHeight;
}

//______________________________________________________________________________
- (void) copyImage : (QuartzImage *) srcImage area : (X11::Rectangle) area withMask : (QuartzImage *) mask clipOrigin : (X11::Point) clipXY toPoint : (X11::Point) dstPoint
{
   using namespace ROOT::MacOSX::X11;

   //Check parameters.
   assert(srcImage != nil && "copyImage:area:withMask:clipOrigin:toPoint:, srcImage parameter is nil");
   assert(srcImage.fImage != nil && "copyImage:area:withMask:clipOrigin:toPoint:, srcImage.fImage is nil");

   if (!AdjustCropArea(srcImage, area)) {
      NSLog(@"QuartzPixmap: -copyImage:srcImage:area:withMask:clipOrigin:toPoint, srcRect and copyRect do not intersect");
      return;
   }
   
   CGImageRef subImage = 0;//RAII not really needed.
   bool needSubImage = false;
   if (area.fX || area.fY || area.fWidth != srcImage.fWidth || area.fHeight != srcImage.fHeight) {
      needSubImage = true;
      subImage = CreateSubImage(srcImage, area);
      if (!subImage) {
         NSLog(@"QuartzPixmap: -copyImage:area:withMask:clipOrigin:toPoint:, subimage creation failed");
         return;
      }
   } else
      subImage = srcImage.fImage;

   //Save context state.
   const Quartz::CGStateGuard stateGuard(fContext);

   if (mask) {
      assert(mask.fImage != nil && "copyImage:area:withMask:clipOrigin:toPoint, mask is not nil, but mask.fImage is nil");
      assert(CGImageIsMask(mask.fImage) && "copyImage:area:withMask:clipOrigin:toPoint, mask.fImage is not a mask");
      clipXY.fY = LocalYROOTToCocoa(self, clipXY.fY + mask.fHeight);//TODO: fix the possible overflow? (though, who can have such images???)
      const CGRect clipRect = CGRectMake(clipXY.fX, clipXY.fY, mask.fWidth, mask.fHeight);
      CGContextClipToMask(fContext, clipRect, mask.fImage);
   }

   dstPoint.fY = LocalYROOTToCocoa(self, dstPoint.fY + area.fHeight);//TODO: fix the possible overflow? (though, who can have such images???)
   const CGRect imageRect = CGRectMake(dstPoint.fX, dstPoint.fY, area.fWidth, area.fHeight);
   CGContextDrawImage(fContext, imageRect, subImage);

   if (needSubImage)
      CGImageRelease(subImage);
}

//______________________________________________________________________________
- (void) copyPixmap : (QuartzPixmap *) srcPixmap area : (X11::Rectangle) area withMask : (QuartzImage *)mask clipOrigin : (X11::Point) clipXY toPoint : (X11::Point) dstPoint
{
   using namespace ROOT::MacOSX::X11;

   assert(srcPixmap != nil && "copyPixmap:area:withMask:clipOrigin:toPoint, srcPixmap parameter is nil");

   if (!AdjustCropArea(srcPixmap, area)) {
      NSLog(@"QuartzPixmap: -copyPixmap:area:withMask:clipOrigin:toPoint, srcRect and copyRect do not intersect");
      return;
   }
   
   const Util::CFScopeGuard<CGImageRef> image([srcPixmap createImageFromPixmap : area]);   
   if (!image.Get())
      return;

   const Quartz::CGStateGuard stateGuard(fContext);
   
   if (mask) {
      assert(mask.fImage != nil && "copyPixmap:area:withMask:clipOrigin:toPoint, mask is not nil, but mask.fImage is nil");
      assert(CGImageIsMask(mask.fImage) && "copyPixmap:area:withMask:clipOrigin:toPoint, mask.fImage is not a mask");
      clipXY.fY = LocalYROOTToCocoa(self, clipXY.fY + mask.fHeight);//TODO: fix the possible overflow? (though, who can have such images???)
      const CGRect clipRect = CGRectMake(clipXY.fX, clipXY.fY, mask.fWidth, mask.fHeight);
      CGContextClipToMask(fContext, clipRect, mask.fImage);
   }
   
   dstPoint.fY = LocalYROOTToCocoa(self, dstPoint.fY + area.fHeight);//TODO: fix the possible overflow? (though, who can have such images???)
   const CGRect imageRect = CGRectMake(dstPoint.fX, dstPoint.fY, area.fWidth, area.fHeight);
   CGContextDrawImage(fContext, imageRect, image.Get());
}

//______________________________________________________________________________
- (void) copy : (NSObject<X11Drawable> *) src area : (X11::Rectangle) area withMask : (QuartzImage *)mask clipOrigin : (X11::Point) origin toPoint : (X11::Point) dstPoint
{
   assert(area.fWidth && area.fHeight && "copy:area:widthMask:clipOrigin:toPoint, empty area to copy");

   if ([src isKindOfClass : [QuartzImage class]]) {
      [self copyImage : (QuartzImage *)src area : area withMask : mask clipOrigin : origin toPoint : dstPoint];
   } else if ([src isKindOfClass : [QuartzPixmap class]]) {
      [self copyPixmap : (QuartzPixmap *)src area : area withMask : mask clipOrigin : origin toPoint : dstPoint];
   } else
      assert(0 && "Can copy only from pixmap or image");
}

//______________________________________________________________________________
- (unsigned char *) readColorBits : (X11::Rectangle) area
{
   assert(area.fWidth && area.fHeight && "readColorBits:, empty area to copy");

   if (!X11::AdjustCropArea(self, area)) {
      NSLog(@"QuartzPixmap: readColorBits:intoBuffer:, src and copy area do not intersect");
      return 0;
   }

   unsigned char *buffer = 0;

   try {
      buffer = new unsigned char[area.fWidth * area.fHeight * 4]();
   } catch (const std::bad_alloc &) {
      NSLog(@"QuartzImage: -readColorBits:, memory allocation failed");
      return 0;
   }


   Util::NSScopeGuard<QuartzPixmap> scaledPixmap;

   if (fScaleFactor > 1) {
      scaledPixmap.Reset([[QuartzPixmap alloc] initWithW : fWidth H : fHeight scaleFactor : 1.]);
      //Ooops, all screwed up!!!
      if (!scaledPixmap.Get()) {
         NSLog(@"QuartzImage: -readColorBits:, can not create scaled pixmap");
         return buffer;//empty buffer.
      }
      
      [scaledPixmap.Get() copy : self area : X11::Rectangle(0, 0, fWidth, fHeight) withMask : nil clipOrigin : X11::Point() toPoint : X11::Point()];
   }

   unsigned char *dstPixel = buffer;

   //fImageData has 4 bytes per pixel.
   //TODO: possible overflows everywhere :(
   const unsigned char *line = fScaleFactor == 1 ? fData + area.fY * fWidth * 4
                               : scaledPixmap.Get()->fData + area.fY * fWidth * 4;

   const unsigned char *srcPixel = line + area.fX * 4;

   for (unsigned i = 0; i < area.fHeight; ++i) {
      for (unsigned j = 0; j < area.fWidth; ++j, srcPixel += 4, dstPixel += 4) {
         dstPixel[0] = srcPixel[0];
         dstPixel[1] = srcPixel[1];
         dstPixel[2] = srcPixel[2];
         dstPixel[3] = srcPixel[3];
      }

      line += fWidth * 4;
      srcPixel = line + area.fX * 4;
   }

   return buffer;
}

//______________________________________________________________________________
- (unsigned char *) fData
{
   return fData;
}

//______________________________________________________________________________
- (void) putPixel : (const unsigned char *) rgb X : (unsigned) x Y : (unsigned) y
{
   //Primitive version of XPutPixel.
   assert(rgb != 0 && "putPixel:X:Y:, rgb parameter is null");
   assert(x < fWidth && "putPixel:X:Y:, x parameter is >= self.fWidth");
   assert(y < fHeight && "putPixel:X:Y:, y parameter is >= self.fHeight");
   
   if (fScaleFactor > 1) {
      //Ooops, and what should I do now???
      const unsigned scaledW = fWidth * fScaleFactor;      
      unsigned char *dst = fData + y * fScaleFactor * scaledW * 4 + x * fScaleFactor * 4;

      for (unsigned i = 0; i < 2; ++i, dst += 4) {
         dst[0] = rgb[0];
         dst[1] = rgb[1];
         dst[2] = rgb[2];
         dst[3] = 255;
      }

      dst -= 8;
      dst += scaledW * 4;

      for (unsigned i = 0; i < 2; ++i, dst += 4) {
         dst[0] = rgb[0];
         dst[1] = rgb[1];
         dst[2] = rgb[2];
         dst[3] = 255;
      }
   } else {
      unsigned char *dst = fData + y * fWidth * 4 + x * 4;
      
      dst[0] = rgb[0];
      dst[1] = rgb[1];
      dst[2] = rgb[2];
      dst[3] = 255;
   }
}

//______________________________________________________________________________
- (void) addPixel : (const unsigned char *) rgb
{
   //Primitive version of XAddPixel.
   assert(rgb != 0 && "addPixel:, rgb parameter is null");
   
   for (unsigned i = 0; i < fHeight; ++i) {
      for (unsigned j = 0; j < fWidth; ++j) {
         fData[i * fWidth * 4 + j * 4] = rgb[0];
         fData[i * fWidth * 4 + j * 4 + 1] = rgb[1];
         fData[i * fWidth * 4 + j * 4 + 2] = rgb[2];
         fData[i * fWidth * 4 + j * 4 + 3] = rgb[3];
      }
   }
}

@end

@implementation QuartzImage {
   unsigned       fWidth;
   unsigned       fHeight;
   CGImageRef     fImage;
   unsigned char *fImageData;
}

@synthesize fIsStippleMask;
@synthesize fID;

//TODO: all these "ctors" were added at different times, not from the beginnning.
//Refactor them to reduce code duplication, where possible.

//______________________________________________________________________________
- (id) initWithW : (unsigned) width H : (unsigned) height data : (unsigned char *) data
{
   assert(width != 0 && "initWithW:H:data:, width parameter is 0");
   assert(height != 0 && "initWithW:H:data:, height parameter is 0");
   assert(data != 0 && "initWithW:H:data:, data parameter is null");

   if (self = [super init]) {
      Util::NSScopeGuard<QuartzImage> selfGuard(self);

      //This w * h * 4 is ONLY for TGCocoa::CreatePixmapFromData.
      //If needed something else, I'll make this code more generic.
      
      unsigned char *dataCopy = 0;
      try {
         dataCopy = new unsigned char[width * height * 4]();
      } catch (const std::bad_alloc &) {
         NSLog(@"QuartzImage: -initWithW:H:data:, memory allocation failed");
         return nil;
      }
      
      std::copy(data, data + width * height * 4, dataCopy);
      Util::ScopedArray<unsigned char> arrayGuard(dataCopy);
   
      fIsStippleMask = NO;
      const CGDataProviderDirectCallbacks providerCallbacks = {0, ROOT_QuartzImage_GetBytePointer, 
                                                               ROOT_QuartzImage_ReleaseBytePointer, 
                                                               ROOT_QuartzImage_GetBytesAtPosition, 0};

      const Util::CFScopeGuard<CGDataProviderRef> provider(CGDataProviderCreateDirect(dataCopy, width * height * 4, &providerCallbacks));
      if (!provider.Get()) {
         NSLog(@"QuartzImage: -initWithW:H:data: CGDataProviderCreateDirect failed");
         return nil;
      }
      
      //RGB - this is only for TGCocoa::CreatePixmapFromData.
      const Util::CFScopeGuard<CGColorSpaceRef> colorSpace(CGColorSpaceCreateDeviceRGB());
      if (!colorSpace.Get()) {
         NSLog(@"QuartzImage: -initWithW:H:data: CGColorSpaceCreateDeviceRGB failed");
         return nil;
      }

      //8 bits per component, 32 bits per pixel, 4 bytes per pixel, kCGImageAlphaLast:
      //all values hardcoded for TGCocoa::CreatePixmapFromData.
      fImage = CGImageCreate(width, height, 8, 32, width * 4, colorSpace.Get(), kCGImageAlphaLast, provider.Get(), 0, false, kCGRenderingIntentDefault);
      
      if (!fImage) {
         NSLog(@"QuartzImage: -initWithW:H:data: CGImageCreate failed");
         return nil;
      }

      selfGuard.Release();
      arrayGuard.Release();

      fWidth = width;
      fHeight = height;
      fImageData = dataCopy;
   }
   
   return self;
}

//______________________________________________________________________________
- (id) initMaskWithW : (unsigned) width H : (unsigned) height bitmapMask : (unsigned char *) mask
{
   assert(width != 0 && "initMaskWithW:H:bitmapMask:, width parameter is zero");
   assert(height != 0 && "initMaskWithW:H:bitmapMask:, height parameter is zero");
   assert(mask != 0 && "initMaskWithW:H:bitmapMask:, mask parameter is null");
   
   if (self = [super init]) {
      Util::NSScopeGuard<QuartzImage> selfGuard(self);
      
      unsigned char *dataCopy = 0;
      try {
         dataCopy = new unsigned char[width * height]();
      } catch (const std::bad_alloc &) {
         NSLog(@"QuartzImage: -initMaskWithW:H:bitmapMask:, memory allocation failed");
         return nil;
      }

      std::copy(mask, mask + width * height, dataCopy);      
      Util::ScopedArray<unsigned char> arrayGuard(dataCopy);
   
      fIsStippleMask = YES;
      const CGDataProviderDirectCallbacks providerCallbacks = {0, ROOT_QuartzImage_GetBytePointer, 
                                                               ROOT_QuartzImage_ReleaseBytePointer, 
                                                               ROOT_QuartzImage_GetBytesAtPosition, 0};
                                                               
                                                               
      const Util::CFScopeGuard<CGDataProviderRef> provider(CGDataProviderCreateDirect(dataCopy, width * height, &providerCallbacks));
      if (!provider.Get()) {
         NSLog(@"QuartzImage: -initMaskWithW:H:bitmapMask: CGDataProviderCreateDirect failed");
         return nil;
      }

      fImage = CGImageMaskCreate(width, height, 8, 8, width, provider.Get(), 0, false);//null -> decode, false -> shouldInterpolate.
      if (!fImage) {
         NSLog(@"QuartzImage: -initMaskWithW:H:bitmapMask:, CGImageMaskCreate failed");
         return nil;
      }
      
      selfGuard.Release();
      arrayGuard.Release();

      fWidth = width;
      fHeight = height;
      fImageData = dataCopy;
   }
   
   return self;
}

//______________________________________________________________________________
- (id) initMaskWithW : (unsigned) width H : (unsigned) height
{
   //Two-step initialization.

   assert(width != 0 && "initMaskWithW:H:, width parameter is zero");
   assert(height != 0 && "initMaskWithW:H:, height parameter is zero");
   
   if (self = [super init]) {
      Util::NSScopeGuard<QuartzImage> selfGuard(self);

      try {
         fImageData = new unsigned char[width * height]();
      } catch (const std::bad_alloc &) {
         NSLog(@"QuartzImage: -initMaskWithW:H:, memory allocation failed");
         return nil;
      }

      fIsStippleMask = YES;
      const CGDataProviderDirectCallbacks providerCallbacks = {0, ROOT_QuartzImage_GetBytePointer, 
                                                               ROOT_QuartzImage_ReleaseBytePointer, 
                                                               ROOT_QuartzImage_GetBytesAtPosition, 0};
                                                               
      const Util::CFScopeGuard<CGDataProviderRef> provider(CGDataProviderCreateDirect(fImageData, width * height, &providerCallbacks));
      if (!provider.Get()) {
         NSLog(@"QuartzImage: -initMaskWithW:H: CGDataProviderCreateDirect failed");
         return nil;
      }

      fImage = CGImageMaskCreate(width, height, 8, 8, width, provider.Get(), 0, false);//null -> decode, false -> shouldInterpolate.
      if (!fImage) {
         NSLog(@"QuartzImage: -initMaskWithW:H:, CGImageMaskCreate failed");
         return nil;
      }
      
      selfGuard.Release();
      
      fWidth = width;
      fHeight = height;
   }
   
   return self;
}

//______________________________________________________________________________
- (id) initFromPixmap : (QuartzPixmap *) pixmap
{
   //Two-step initialization.
   assert(pixmap != nil && "initFromPixmap:, pixmap parameter is nil");
   assert(pixmap.fWidth != 0 && "initFromPixmap:, pixmap width is zero");
   assert(pixmap.fHeight != 0 && "initFromPixmap:, pixmap height is zero");

   return [self initWithW : pixmap.fWidth H : pixmap.fHeight data : pixmap.fData];
}

//______________________________________________________________________________
- (id) initFromImage : (QuartzImage *) image
{
   assert(image != nil && "initFromImage:, image parameter is nil");
   assert(image.fWidth != 0 && "initFromImage:, image width is 0");
   assert(image.fHeight != 0 && "initFromImage:, image height is 0");
   assert(image.fIsStippleMask == NO && "initFromImage:, image is a stipple mask, not implemented");
   
   return [self initWithW : image.fWidth H : image.fHeight data : image->fImageData];
}

//______________________________________________________________________________
- (id) initFromImageFlipped : (QuartzImage *) image
{
   assert(image != nil && "initFromImageFlipped:, image parameter is nil");
   assert(image.fWidth != 0 && "initFromImageFlipped:, image width is 0");
   assert(image.fHeight != 0 && "initFromImageFlipped:, image height is 0");
   
   const unsigned bpp = image.fIsStippleMask ? 1 : 4;
   
   if (self = [super init]) {
      const unsigned width = image.fWidth;
      const unsigned height = image.fHeight;

      Util::NSScopeGuard<QuartzImage> selfGuard(self);

      unsigned char *dataCopy = 0;
      try {
         dataCopy = new unsigned char[width * height * bpp]();
      } catch (const std::bad_alloc &) {
         NSLog(@"QuartzImage: -initFromImageFlipped:, memory allocation failed");
         return nil;
      }

      const unsigned lineSize = bpp * width;
      for (unsigned i = 0; i < height; ++i) {
         const unsigned char *sourceLine = image->fImageData + lineSize * (height - 1 - i);
         unsigned char *dstLine = dataCopy + i * lineSize;
         std::copy(sourceLine, sourceLine + lineSize, dstLine);
      }
      
      Util::ScopedArray<unsigned char> arrayGuard(dataCopy);

      const CGDataProviderDirectCallbacks providerCallbacks = {0, ROOT_QuartzImage_GetBytePointer, 
                                                               ROOT_QuartzImage_ReleaseBytePointer,
                                                               ROOT_QuartzImage_GetBytesAtPosition, 0};
   
      if (bpp == 1) {
         fIsStippleMask = YES;

         const Util::CFScopeGuard<CGDataProviderRef> provider(CGDataProviderCreateDirect(dataCopy, width * height, &providerCallbacks));
         if (!provider.Get()) {
            NSLog(@"QuartzImage: -initFromImageFlipped:, CGDataProviderCreateDirect failed");
            return nil;
         }

         fImage = CGImageMaskCreate(width, height, 8, 8, width, provider.Get(), 0, false);//null -> decode, false -> shouldInterpolate.
         if (!fImage) {
            NSLog(@"QuartzImage: -initFromImageFlipped:, CGImageMaskCreate failed");
            return nil;
         }
      } else {
         fIsStippleMask = NO;
      
         const Util::CFScopeGuard<CGDataProviderRef> provider(CGDataProviderCreateDirect(dataCopy, width * height * 4, &providerCallbacks));
         if (!provider.Get()) {
            NSLog(@"QuartzImage: -initFromImageFlipped:, CGDataProviderCreateDirect failed");
            return nil;
         }
      
         const Util::CFScopeGuard<CGColorSpaceRef> colorSpace(CGColorSpaceCreateDeviceRGB());
         if (!colorSpace.Get()) {
            NSLog(@"QuartzImage: -initFromImageFlipped:, CGColorSpaceCreateDeviceRGB failed");
            return nil;
         }

         //8 bits per component, 32 bits per pixel, 4 bytes per pixel, kCGImageAlphaLast:
         //all values hardcoded for TGCocoa::CreatePixmapFromData.
         fImage = CGImageCreate(width, height, 8, 32, width * 4, colorSpace.Get(), kCGImageAlphaLast, provider.Get(), 0, false, kCGRenderingIntentDefault);
         if (!fImage) {
            NSLog(@"QuartzImage: -initFromImageFlipped:, CGImageCreate failed");
            return nil;
         }
      }
      
      selfGuard.Release();
      arrayGuard.Release();

      fWidth = width;
      fHeight = height;
      fImageData = dataCopy;
   }
   
   return self;
}

//______________________________________________________________________________
- (void) dealloc
{
   if (fImage) {
      CGImageRelease(fImage);
      delete [] fImageData;
   }
   
   [super dealloc];
}

//______________________________________________________________________________
- (BOOL) isRectInside : (X11::Rectangle) area
{
   if (area.fX < 0 || (unsigned)area.fX >= fWidth)
      return NO;
   if (area.fY < 0 || (unsigned)area.fY >= fHeight)
      return NO;
   if (area.fWidth > fWidth || !area.fWidth)
      return NO;
   if (area.fHeight > fHeight || !area.fHeight)
      return NO;
   
   return YES;
}

//______________________________________________________________________________
- (unsigned char *) readColorBits : (X11::Rectangle) area
{
   assert([self isRectInside : area] == YES && "readColorBits: bad area parameter");
   //Image, bitmap - they all must be converted to ARGB (bitmap) or BGRA (image) (for libAfterImage).
   unsigned char *buffer = 0;
   
   try {
      buffer = new unsigned char[area.fWidth * area.fHeight * 4]();
   } catch (const std::bad_alloc &) {
      NSLog(@"QuartzImage: -readColorBits:, memory allocation failed");
      return 0;
   }

   unsigned char *dstPixel = buffer;

   if (CGImageIsMask(fImage)) {
      //fImageData has 1 byte per pixel.
      const unsigned char *line = fImageData + area.fY * fWidth;
      const unsigned char *srcPixel =  line + area.fX;

      for (unsigned i = 0; i < area.fHeight; ++i) {
         for (unsigned j = 0; j < area.fWidth; ++j, ++srcPixel, dstPixel += 4) {
            if (!srcPixel[0])
               dstPixel[0] = 255;//can be 1 or anything different from 0.
         }
         
         line += fWidth;
         srcPixel = line + area.fX;
      }

   } else {
      //fImageData has 4 bytes per pixel.
      const unsigned char *line = fImageData + area.fY * fWidth * 4;
      const unsigned char *srcPixel = line + area.fX * 4;
      
      for (unsigned i = 0; i < area.fHeight; ++i) {
         for (unsigned j = 0; j < area.fWidth; ++j, srcPixel += 4, dstPixel += 4) {
            dstPixel[0] = srcPixel[2];
            dstPixel[1] = srcPixel[1];
            dstPixel[2] = srcPixel[0];
            dstPixel[3] = srcPixel[3];
         }

         line += fWidth * 4;
         srcPixel = line + area.fX * 4;
      }
      
      return buffer;
   }
   
   return buffer;
}

//______________________________________________________________________________
- (BOOL) fIsPixmap
{
   return YES;
}

//______________________________________________________________________________
- (BOOL) fIsOpenGLWidget
{
   return NO;
}

//______________________________________________________________________________
- (unsigned) fWidth
{
   return fWidth;
}

//______________________________________________________________________________
- (unsigned) fHeight
{
   return fHeight;
}

//______________________________________________________________________________
- (CGImageRef) fImage
{
   return fImage;
}

@end

namespace ROOT {
namespace MacOSX {
namespace X11 {

//______________________________________________________________________________
CGImageRef CreateSubImage(QuartzImage *image, const Rectangle &area)
{
   assert(image != nil && "CreateSubImage, image parameter is nil");

   const CGRect subImageRect = CGRectMake(area.fX, area.fY, area.fHeight, area.fWidth);
   return CGImageCreateWithImageInRect(image.fImage, subImageRect);
}

namespace {

//Now, close your eyes and open them at the end of this block. :)
//Sure, this can be done easy, but I hate to convert between negative signed integers and
//unsigned integers and the other way, so I have this implementation (integers will be always
//positive and they obviously fit into unsigned integers).

typedef std::pair<int, unsigned> range_type;

//______________________________________________________________________________
bool FindOverlapSameSigns(const range_type &left, const range_type &right, range_type &intersection)
{
   //"Same" means both xs are non-negative, or both are negative.
   //left.x <= right.x.
   const unsigned dX(right.first - left.first);//diff fits into the positive range of int.
   //No intersection.
   if (dX >= left.second)
      return false;
   //Find an intersection.
   intersection.first = right.first;
   intersection.second = std::min(right.second, left.second - dX);//left.second is always > dX.

   return true;
}

//______________________________________________________________________________
bool FindOverlapDifferentSigns(const range_type &left, const range_type &right, range_type &intersection)
{
   //x2 - x1 can overflow.
   //Left.x is negative, right.x is non-negative (0 included).
   const unsigned signedMinAbs(std::numeric_limits<unsigned>::max() / 2 + 1);
   
   if (left.first == std::numeric_limits<int>::min()) {//hehehe
      if (left.second <= signedMinAbs)
         return false;
      
      if (left.second - signedMinAbs <= unsigned(right.first))
         return false;
      
      intersection.first = right.first;
      intersection.second = std::min(right.second, left.second - signedMinAbs - unsigned(right.first));
   } else {
      const unsigned leftXAbs(-left.first);//-left.first can't overflow.
      if (leftXAbs >= left.second)
         return false;
      
      if (left.second - leftXAbs <= unsigned(right.first))
         return false;
      
      intersection.first = right.first;
      intersection.second = std::min(right.second, left.second - leftXAbs - unsigned(right.first));
   }

   return true;
}

//______________________________________________________________________________
bool FindOverlap(const range_type &range1, const range_type &range2, range_type &intersection)
{
   range_type left;
   range_type right;
   
   if (range1.first < range2.first) {
      left = range1;
      right = range2;
   } else {
      left = range2;
      right = range1;
   }

   if (left.first < 0)
      return right.first < 0 ? FindOverlapSameSigns(left, right, intersection) : FindOverlapDifferentSigns(left, right, intersection);

   return FindOverlapSameSigns(left, right, intersection);
}

}

//______________________________________________________________________________
bool AdjustCropArea(const Rectangle &srcRect, Rectangle &cropArea)
{
   //Find rects intersection.
   range_type xIntersection;
   if (!FindOverlap(range_type(srcRect.fX, srcRect.fWidth), range_type(cropArea.fX, cropArea.fWidth), xIntersection))
      return false;
   
   range_type yIntersection;
   if (!FindOverlap(range_type(srcRect.fY, srcRect.fHeight), range_type(cropArea.fY, cropArea.fHeight), yIntersection))
      return false;
   
   cropArea.fX = xIntersection.first;
   cropArea.fWidth = xIntersection.second;
   
   cropArea.fY = yIntersection.first;
   cropArea.fHeight = yIntersection.second;
   
   return true;
}

//______________________________________________________________________________
bool AdjustCropArea(QuartzImage *srcImage, Rectangle &cropArea)
{
   assert(srcImage != nil && "AdjustCropArea, srcImage parameter is nil");
   assert(srcImage.fImage != nil && "AdjustCropArea, srcImage.fImage is nil");

   return AdjustCropArea(X11::Rectangle(0, 0, srcImage.fWidth, srcImage.fHeight), cropArea);
}

//______________________________________________________________________________
bool AdjustCropArea(QuartzImage *srcImage, NSRect &cropArea)
{
   assert(srcImage != nil && "AdjustCropArea, srcImage parameter is nil");
   assert(srcImage.fImage != 0 && "AdjustCropArea, srcImage.fImage is null");
   
   const Rectangle srcRect(0, 0, srcImage.fWidth, srcImage.fHeight);
   Rectangle dstRect(int(cropArea.origin.x), int(cropArea.origin.y), unsigned(cropArea.size.width), unsigned(cropArea.size.height));
   
   if (AdjustCropArea(srcRect, dstRect)) {
      cropArea.origin.x = dstRect.fX;
      cropArea.origin.y = dstRect.fY;
      cropArea.size.width = dstRect.fWidth;
      cropArea.size.height = dstRect.fHeight;

      return true;
   }
   
   return false;
}

//______________________________________________________________________________
bool AdjustCropArea(QuartzPixmap *srcPixmap, X11::Rectangle &cropArea)
{
   assert(srcPixmap != nil && "AdjustCropArea, srcPixmap parameter is nil");

   return AdjustCropArea(X11::Rectangle(0, 0, srcPixmap.fWidth, srcPixmap.fHeight), cropArea);
}

//______________________________________________________________________________
bool TestBitmapBit(const unsigned char *bitmap, unsigned w, unsigned i, unsigned j)
{
   //Test if a bit (i,j) is set in a bitmap (w, h).
   
   //Code in ROOT's GUI suggests, that byte is octet.
   assert(bitmap != 0 && "TestBitmapBit, bitmap parameter is null");
   assert(w != 0 && "TestBitmapBit, w parameter is 0");
   assert(i < w && "TestBitmapBit, i parameter is >= w");
   
   const unsigned bytesPerLine = (w + 7) / 8;
   const unsigned char *line = bitmap + j * bytesPerLine;
   const unsigned char byteValue = line[i / 8];
   
   return byteValue & (1 << (i % 8));
}

//______________________________________________________________________________
void FillPixmapBuffer(const unsigned char *bitmap, unsigned width, unsigned height, ULong_t foregroundPixel, ULong_t backgroundPixel, unsigned depth, unsigned char *imageData)
{
   assert(bitmap != 0 && "FillPixmapBuffer, bitmap parameter is null");
   assert(width != 0 && "FillPixmapBuffer, width parameter is 0");
   assert(height != 0 && "FillPixmapBuffer, height parameter is 0");
   assert(imageData != 0 && "FillPixmapBuffer, imageData parameter is null");

   if (depth > 1) {
      unsigned char foregroundColor[4] = {};
      PixelToRGB(foregroundPixel, foregroundColor);
      unsigned char backgroundColor[4] = {};
      PixelToRGB(backgroundPixel, backgroundColor);

      for (unsigned j = 0; j < height; ++j) {
         const unsigned line = j * width * 4;
         for (unsigned i = 0; i < width; ++i) {
            const unsigned pixel = line + i * 4;
            
            if (TestBitmapBit(bitmap, width, i, j)) {
               //Foreground color.
               imageData[pixel] = foregroundColor[0];
               imageData[pixel + 1] = foregroundColor[1];
               imageData[pixel + 2] = foregroundColor[2];
            } else {
               imageData[pixel] = backgroundColor[0];
               imageData[pixel + 1] = backgroundColor[1];
               imageData[pixel + 2] = backgroundColor[2];            
            }
            
            imageData[pixel + 3] = 255;
         }
      }
   } else {
      for (unsigned j = 0; j < height; ++j) {
         const unsigned line = j * width;
         for (unsigned i = 0; i < width; ++i) {
            const unsigned pixel = line + i;
            if (TestBitmapBit(bitmap, width, i, j))
               imageData[pixel] = 0;
            else
               imageData[pixel] = 255;//mask out pixel.
         }
      }   
   }
}

}//X11
}//MacOSX
}//ROOT
