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

#import <cstdlib>
#import <cassert>
#import <cstddef>

#import "QuartzWindow.h"//TODO: Move conversion functions from QuartzWindow to X11Coords or something like this.
#import "QuartzPixmap.h"
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

@implementation QuartzPixmap {
@private
   unsigned       fWidth;
   unsigned       fHeight;
   unsigned char *fData;
   CGContextRef   fContext;
}

@synthesize fID;

//______________________________________________________________________________
- (id) initWithW : (unsigned) width H : (unsigned) height
{
   if (self = [super init]) {
      fWidth = 0;
      fHeight = 0;
      fData = 0;
      
      if ([self resizeW : width H : height])
         return self;
   }

   //Two step initialization:
   //1. p = [QuartzPixmap alloc];
   //2. p1 = [p initWithW : w H : h];
   // if (!p1) [p release];
   return nil;
}

//______________________________________________________________________________
- (void) dealloc
{
   if (fContext)
      CGContextRelease(fContext);
   if (fData)
      std::free(fData);

   [super dealloc];
}

//______________________________________________________________________________
- (BOOL) resizeW : (unsigned) width H : (unsigned) height
{
   assert(width > 0 && "Pixmap width must be positive");
   assert(height > 0 && "Pixmap height must be positive");

   unsigned char *memory = (unsigned char *)malloc(width * height * 4);//[0]
   if (!memory) {
      assert(0 && "resizeW:H:, malloc failed");
      return NO;
   }

   CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();//[1]
   if (!colorSpace) {
      assert(0 && "resizeW:H:, CGColorSpaceCreateDeviceRGB failed");
      std::free(memory);
      return NO;
   }

   //
   CGContextRef ctx = CGBitmapContextCreateWithData(memory, width, height, 8, width * 4, colorSpace, kCGImageAlphaPremultipliedLast, NULL, 0);
   if (!ctx) {
      assert(0 && "resizeW:H:, CGBitmapContextCreateWidthData failed");
      CGColorSpaceRelease(colorSpace);
      std::free(memory);
      return NO;
   }

   if (fContext) {
      //New context was created OK, we can release now the old one.
      CGContextRelease(fContext);//[2]
   }
   
   if (fData) {
      //Release old memory.
      std::free(fData);
   }

   //Size to be used later - to identify,
   //if we really have to resize.
   fWidth = width;
   fHeight = height;
   fData = memory;

   fContext = ctx;//[2]

   CGColorSpaceRelease(colorSpace);//[1]

   return YES;

}

//______________________________________________________________________________
- (CGImageRef) createImageFromPixmap
{
   Rectangle_t imageRect = {};
   imageRect.fX = 0;
   imageRect.fY = 0;
   imageRect.fWidth = fWidth;
   imageRect.fHeight = fHeight;
   
   return [self createImageFromPixmap : imageRect];
}

//______________________________________________________________________________
- (CGImageRef) createImageFromPixmap : (Rectangle_t) cropArea
{
   //Crop area must be valid and adjusted by caller.
   assert(cropArea.fX >= 0 && "createImageFromPixmap:, cropArea.fX is negative");
   assert(cropArea.fY >= 0 && "createImageFromPixmap:, cropArea.fY is negative");
   assert(cropArea.fWidth <= fWidth && "createImageFromPixmap:, bad cropArea.fWidth");
   assert(cropArea.fHeight <= fHeight && "createImageFromPixmap:, bad cropArea.fHeight");

   //
   const CGDataProviderDirectCallbacks providerCallbacks = {0, ROOT_QuartzImage_GetBytePointer, 
                                                            ROOT_QuartzImage_ReleaseBytePointer, 
                                                            ROOT_QuartzImage_GetBytesAtPosition, 0};

   
   CGDataProviderRef provider = CGDataProviderCreateDirect(fData, fWidth * fHeight * 4, &providerCallbacks);
   if (!provider) {
      NSLog(@"QuartzPixmap: -pixmapToImage, CGDataProviderCreateDirect failed");
      return 0;
   }

   //RGB - this is only for TGCocoa::CreatePixmapFromData.
   CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
   if (!colorSpace) {
      NSLog(@"QuartzPixmap: -pixmapToImage, CGColorSpaceCreateDeviceRGB failed");
      CGDataProviderRelease(provider);
      return 0;
   }
      
   //8 bits per component, 32 bits per pixel, 4 bytes per pixel, kCGImageAlphaLast:
   //all values hardcoded for TGCocoa.
   CGImageRef image = CGImageCreate(cropArea.fWidth, cropArea.fHeight, 8, 32, fWidth * 4, colorSpace, kCGImageAlphaPremultipliedLast, provider, 0, false, kCGRenderingIntentDefault);
   CGColorSpaceRelease(colorSpace);
   CGDataProviderRelease(provider);
   
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
- (void) copyImage : (QuartzImage *) srcImage area : (Rectangle_t) area withMask : (QuartzImage *) mask clipOrigin : (Point_t) clipXY toPoint : (Point_t) dstPoint
{
   using namespace ROOT::MacOSX::X11;

   //Check parameters.
   assert(srcImage != nil && "copyImage:area:withMask:clipOrigin:toPoint:, srcImage parameter is nil");
   assert(srcImage.fImage != nil && "copyImage:area:withMask:clipOrigin:toPoint:, srcImage.fImage is nil");

   if (!AdjustCropArea(srcImage, area)) {
      NSLog(@"QuartzPixmap: -copyImage:srcImage:area:withMask:clipOrigin:toPoint, srcRect and copyRect do not intersect");
      return;
   }
   
   CGImageRef subImage = 0;
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
   CGContextSaveGState(fContext);

   CGContextTranslateCTM(fContext, 0., fHeight);
   CGContextScaleCTM(fContext, 1., -1.);


   if (mask) {
      assert(mask.fImage != nil && "copyImage:area:withMask:clipOrigin:toPoint, mask is not nil, but mask.fImage is nil");
      assert(CGImageIsMask(mask.fImage) && "copyImage:area:withMask:clipOrigin:toPoint, mask.fImage is not a mask");
      clipXY.fY = LocalYROOTToCocoa(self, clipXY.fY + mask.fHeight);
      const CGRect clipRect = CGRectMake(clipXY.fX, clipXY.fY, mask.fWidth, mask.fHeight);
      CGContextClipToMask(fContext, clipRect, mask.fImage);
   }

   dstPoint.fY = LocalYROOTToCocoa(self, dstPoint.fY + area.fHeight);
   const CGRect imageRect = CGRectMake(dstPoint.fX, dstPoint.fY, area.fWidth, area.fHeight);
   CGContextDrawImage(fContext, imageRect, subImage);
   //Restore context state.
   CGContextRestoreGState(fContext);

   if (needSubImage)
      CGImageRelease(subImage);
}

//______________________________________________________________________________
- (void) copyPixmap : (QuartzPixmap *) srcPixmap area : (Rectangle_t) area withMask : (QuartzImage *)mask clipOrigin : (Point_t) clipXY toPoint : (Point_t) dstPoint
{
   using namespace ROOT::MacOSX::X11;

   assert(srcPixmap != nil && "copyPixmap:area:withMask:clipOrigin:toPoint, srcPixmap parameter is nil");

   if (!AdjustCropArea(srcPixmap, area)) {
      NSLog(@"QuartzPixmap: -copyPixmap:area:withMask:clipOrigin:toPoint, srcRect and copyRect do not intersect");
      return;
   }
   
   CGImageRef image = [srcPixmap createImageFromPixmap : area];
   
   if (!image)
      return;
      
   CGContextSaveGState(fContext);
   CGContextTranslateCTM(fContext, 0., fHeight);
   CGContextScaleCTM(fContext, 1., -1.);
   
   if (mask) {
      assert(mask.fImage != nil && "copyPixmap:area:withMask:clipOrigin:toPoint, mask is not nil, but mask.fImage is nil");
      assert(CGImageIsMask(mask.fImage) && "copyPixmap:area:withMask:clipOrigin:toPoint, mask.fImage is not a mask");
      clipXY.fY = LocalYROOTToCocoa(self, clipXY.fY + mask.fHeight);
      const CGRect clipRect = CGRectMake(clipXY.fX, clipXY.fY, mask.fWidth, mask.fHeight);
      CGContextClipToMask(fContext, clipRect, mask.fImage);
   }
   
   dstPoint.fY = LocalYROOTToCocoa(self, dstPoint.fY + area.fHeight);
   const CGRect imageRect = CGRectMake(dstPoint.fX, dstPoint.fY, area.fWidth, area.fHeight);
   CGContextDrawImage(fContext, imageRect, image);
   //Restore context state.
   CGContextRestoreGState(fContext);

   CGImageRelease(image);
}

//______________________________________________________________________________
- (void) copy : (NSObject<X11Drawable> *) src area : (Rectangle_t) area withMask : (QuartzImage *)mask clipOrigin : (Point_t) origin toPoint : (Point_t) dstPoint
{
   if ([src isKindOfClass : [QuartzImage class]]) {
      [self copyImage : (QuartzImage *)src area : area withMask : mask clipOrigin : origin toPoint : dstPoint];
   } else if ([src isKindOfClass : [QuartzPixmap class]]) {
      [self copyPixmap : (QuartzPixmap *)src area : area withMask : mask clipOrigin : origin toPoint : dstPoint];
   } else
      assert(0 && "Can copy only from pixmap or image");
}

//______________________________________________________________________________
- (unsigned char *) fData
{
   return fData;
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

//______________________________________________________________________________
- (id) initWithW : (unsigned) width H : (unsigned) height data : (unsigned char *)data
{
   //Two step initialization. If the second step (initWithW:....) fails, user must call release 
   //(after he checked the result of init call).

   assert(width != 0 && "initWithW:H:data:, width parameter is 0");
   assert(height != 0 && "initWithW:H:data:, height parameter is 0");
   assert(data != 0 && "initWithW:H:data:, data parameter is null");

   if (self = [super init]) {
      fIsStippleMask = NO;
      const CGDataProviderDirectCallbacks providerCallbacks = {0, ROOT_QuartzImage_GetBytePointer, 
                                                               ROOT_QuartzImage_ReleaseBytePointer, 
                                                               ROOT_QuartzImage_GetBytesAtPosition, 0};

      //This w * h * 4 is ONLY for TGCocoa::CreatePixmapFromData.
      //If needed something else, I'll make this code more generic.
      CGDataProviderRef provider = CGDataProviderCreateDirect(data, width * height * 4, &providerCallbacks);
      if (!provider) {
         NSLog(@"QuartzPixmap: -initWithW:H:data: CGDataProviderCreateDirect failed");
         return nil;
      }
      
      //RGB - this is only for TGCocoa::CreatePixmapFromData.
      CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
      if (!colorSpace) {
         NSLog(@"QuartzPixmap: -initWithW:H:data: CGColorSpaceCreateDeviceRGB failed");
         CGDataProviderRelease(provider);
         return nil;
      }
      
      //8 bits per component, 32 bits per pixel, 4 bytes per pixel, kCGImageAlphaLast:
      //all values hardcoded for TGCocoa::CreatePixmapFromData.
      fImage = CGImageCreate(width, height, 8, 32, width * 4, colorSpace, kCGImageAlphaLast, provider, 0, false, kCGRenderingIntentDefault);
      CGColorSpaceRelease(colorSpace);
      CGDataProviderRelease(provider);
      
      if (!fImage) {
         NSLog(@"QuartzPixmap: -initWithW:H:data: CGImageCreate failed");
         return nil;
      }

      fWidth = width;
      fHeight = height;

      fImageData = data;

      return self;
   }
   
   return nil;
}

//______________________________________________________________________________
- (id) initMaskWithW : (unsigned) width H : (unsigned) height bitmapMask : (unsigned char *)mask
{
   assert(width > 0 && "initMaskWithW:H:bitmapMask:, width parameter is zero");
   assert(height > 0 && "initMaskWithW:H:bitmapMask:, height parameter is zero");
   assert(mask != 0 && "initMaskWithW:H:bitmapMask:, mask parameter is null");
   
   if (self = [super init]) {
      fIsStippleMask = YES;
      const CGDataProviderDirectCallbacks providerCallbacks = {0, ROOT_QuartzImage_GetBytePointer, 
                                                               ROOT_QuartzImage_ReleaseBytePointer, 
                                                               ROOT_QuartzImage_GetBytesAtPosition, 0};
      CGDataProviderRef provider = CGDataProviderCreateDirect(mask, width * height, &providerCallbacks);
      if (!provider) {
         NSLog(@"QuartzPixmap: -initMaskWithW:H:bitmapMask: CGDataProviderCreateDirect failed");
         return nil;
      }

      fImage = CGImageMaskCreate(width, height, 8, 8, width, provider, 0, false);//null -> decode, false -> shouldInterpolate.
      CGDataProviderRelease(provider);

      if (!fImage) {
         NSLog(@"QuartzPixmap: -initMaskWithW:H:bitmapMask:, CGImageMaskCreate failed");
         return nil;
      }
      
      fWidth = width;
      fHeight = height;

      fImageData = mask;
      
      return self;
   }
   
   return nil;
}

//______________________________________________________________________________
- (id) initMaskWithW : (unsigned) width H : (unsigned) height
{
   assert(width > 0 && "initMaskWithW:H:, width parameter is zero");
   assert(height > 0 && "initMaskWithW:H:, height parameter is zero");
   
   if (self = [super init]) {
      fImageData = new unsigned char[width * height];
      fIsStippleMask = YES;
      const CGDataProviderDirectCallbacks providerCallbacks = {0, ROOT_QuartzImage_GetBytePointer, 
                                                               ROOT_QuartzImage_ReleaseBytePointer, 
                                                               ROOT_QuartzImage_GetBytesAtPosition, 0};
      CGDataProviderRef provider = CGDataProviderCreateDirect(fImageData, width * height, &providerCallbacks);
      if (!provider) {
         NSLog(@"QuartzPixmap: -initMaskWithW:H: CGDataProviderCreateDirect failed");
         delete [] fImageData;
         fImageData = 0;
         return nil;
      }

      fImage = CGImageMaskCreate(width, height, 8, 8, width, provider, 0, false);//null -> decode, false -> shouldInterpolate.
      CGDataProviderRelease(provider);

      if (!fImage) {
         NSLog(@"QuartzPixmap: -initMaskWithW:H:, CGImageMaskCreate failed");
         delete [] fImageData;
         fImageData = 0;
         return nil;
      }
      
      fWidth = width;
      fHeight = height;
      
      return self;
   }
   
   return nil;
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
- (void) clearMask
{
   assert(fIsStippleMask == YES && "-clearMask, called for non-mask image");
   
   for (unsigned i = 0, e = fWidth * fHeight; i < e; ++i)
      fImageData[i] = 0;//All pixels are ok.
}

//______________________________________________________________________________
- (void) maskOutPixels : (NSRect) maskedArea
{
   assert(fIsStippleMask == YES && "-maskOutPixels, called for non-mask image");
   assert(fImageData != 0 && "-maskOutPixels, image was not initialized");
   
   const int iStart = std::max(0, int(maskedArea.origin.x));
   const int iEnd = std::min(int(fWidth), int(maskedArea.size.width) + iStart);
   
   //Note about j: as soon as QuartzView is flipped, orde of pixel lines is changed here.
   const int jStart = int(fHeight) - std::min(int(fHeight), int(maskedArea.origin.y + maskedArea.size.height));
   const int jEnd = std::min(int(fHeight), int(jStart + maskedArea.size.height));
   
   for (int j = jStart; j < jEnd; ++j) {
      unsigned char *line = fImageData + j * fWidth;
      for (int i = iStart; i < iEnd; ++i)
         line[i] = 255;
   }
}

//______________________________________________________________________________
- (BOOL) isRectInside : (Rectangle_t) area
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
- (unsigned char *) readColorBits : (Rectangle_t) area
{
   assert([self isRectInside : area] == YES && "readColorBits: bad area parameter");
   //Image, bitmap - they all must be converted to ARGB (bitmap) or BGRA (image) (for libAfterImage).
   unsigned char *buffer = new unsigned char[area.fWidth * area.fHeight * 4]();
   unsigned char *dstPixel = buffer;

   if (CGImageIsMask(fImage)) {
      //fImageData has 1 byte per pixel.
      const unsigned char *line = fImageData + area.fY * fWidth;
      const unsigned char *srcPixel =  line + area.fX;

      for (UShort_t i = 0; i < area.fHeight; ++i) {
         for (UShort_t j = 0; j < area.fWidth; ++j, ++srcPixel, dstPixel += 4) {
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
      
      for (UShort_t i = 0; i < area.fHeight; ++i) {
         for (UShort_t j = 0; j < area.fWidth; ++j, srcPixel += 4, dstPixel += 4) {
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
CGImageRef CreateSubImage(QuartzImage *image, const Rectangle_t &area)
{
   assert(image != nil && "CreateSubImage, image parameter is nil");

   const CGRect subImageRect = CGRectMake(area.fX, area.fY, area.fHeight, area.fWidth);
   return CGImageCreateWithImageInRect(image.fImage, subImageRect);
}

//______________________________________________________________________________
bool AdjustCropArea(const Rectangle_t &srcRect, Rectangle_t &cropArea)
{
   //First, find cases, when srcRect and cropArea do not intersect.
   if (cropArea.fX >= srcRect.fX + int(srcRect.fWidth))
      return false;//No intersection: crop on the right of source.
   if (cropArea.fX + int(cropArea.fWidth) <= srcRect.fX)
      return false;//No intersection: crop on the left of source.
      
   if (cropArea.fY >= srcRect.fY + int(srcRect.fHeight))
      return false;//No intersection: crop is above the source.
   if (cropArea.fY + int(cropArea.fHeight) <= srcRect.fY)
      return false;//No intersection: crop is under the source.
      
   //Intersection exists, set crop area to this intersection.
   if (cropArea.fX < srcRect.fX) {
      cropArea.fWidth = std::min(int(srcRect.fWidth), int(cropArea.fWidth) - int(srcRect.fX - cropArea.fX));
      cropArea.fX = srcRect.fX;
   } else
      cropArea.fWidth = std::min(int(srcRect.fWidth) - int(cropArea.fX - srcRect.fX), int(cropArea.fWidth));
      
   if (cropArea.fY < srcRect.fY) {
      cropArea.fHeight = std::min(int(srcRect.fHeight), int(cropArea.fHeight) - int(srcRect.fY - cropArea.fY));
      cropArea.fY = srcRect.fY;
   } else
      cropArea.fHeight = std::min(int(srcRect.fHeight) - int(cropArea.fY - srcRect.fY), int(cropArea.fHeight));
   
   return true;
}

//______________________________________________________________________________
bool AdjustCropArea(QuartzImage *srcImage, Rectangle_t &cropArea)
{
   assert(srcImage != nil && "AdjustCropArea, srcImage parameter is nil");
   assert(srcImage.fImage != nil && "AdjustCropArea, srcImage.fImage is nil");
   
   Rectangle_t srcRect = {};
   srcRect.fX = 0, srcRect.fY = 0;
   srcRect.fWidth = srcImage.fWidth;
   srcRect.fHeight = srcImage.fHeight;
   
   return AdjustCropArea(srcRect, cropArea);
}

//______________________________________________________________________________
bool AdjustCropArea(QuartzPixmap *srcPixmap, Rectangle_t &cropArea)
{
   assert(srcPixmap != nil && "AdjustCropArea, srcPixmap parameter is nil");

   Rectangle_t srcRect = {};
   srcRect.fX = 0, srcRect.fY = 0;
   srcRect.fWidth = srcPixmap.fWidth;
   srcRect.fHeight = srcPixmap.fHeight;
   
   return AdjustCropArea(srcRect, cropArea);
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
      X11::PixelToRGB(foregroundPixel, foregroundColor);
      unsigned char backgroundColor[4] = {};
      X11::PixelToRGB(backgroundPixel, backgroundColor);

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

}
}
}
