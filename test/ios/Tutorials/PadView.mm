#import <cstddef>
#import <vector>
#import <cmath>
#import <new>

#import <CoreGraphics/CoreGraphics.h>
#import <CoreGraphics/CGContext.h>

#import "SelectionView.h"
#import "PadView.h"

//C++ code (ROOT's ios module)
#import "IOSPad.h"

@interface PadView () {
   ROOT::iOS::Pad *pad;

   float scaleFactor;
   SelectionView *selectionView;
   
   BOOL processPan;
   BOOL processTap;
}

@end

@implementation PadView

//_________________________________________________________________
- (id) initWithFrame : (CGRect) frame forPad : (ROOT::iOS::Pad*) pd
{
   self = [super initWithFrame : frame];
   if (self) {
      assert(pd != nullptr && "initWithFrame:forPad:, parameter 'pd' is null");
      //Initialize C++ objects here.
      pad = pd;

      scaleFactor = frame.size.width / 640.f;
   }

   return self;
}

//_________________________________________________________________
- (void) drawRect : (CGRect)rect
{
   // Drawing code
   assert(pad != nullptr && "drawRect:, pad is null");
   
   CGContextRef ctx = UIGraphicsGetCurrentContext();

   CGContextClearRect(ctx, rect);

   CGContextTranslateCTM(ctx, 0.f, rect.size.height);
   CGContextScaleCTM(ctx, 1.f, -1.f);

   CGContextScaleCTM(ctx, scaleFactor, scaleFactor);

   pad->cd();
   pad->SetContext(ctx);
   pad->Paint();
}

//_________________________________________________________________
- (void) clearPad
{
   assert(pad != nullptr && "clearPad, pad is null");

   pad->Clear();
}

//_________________________________________________________________
- (void) handlePanGesture : (UIPanGestureRecognizer *) panGesture
{
   if (!processPan)
      return;

   assert(pad != nullptr && "handlePanGesture:, pad is null");

   const CGPoint p = [panGesture locationInView : self];
   [selectionView setPad : pad];
   [selectionView setShowRotation : YES];
   
   if (panGesture.state == UIGestureRecognizerStateBegan) {
      selectionView.hidden = NO;
      [selectionView setEvent : kButton1Down atX : p.x andY : p.y];
      [selectionView setNeedsDisplay];
   } else if (panGesture.state == UIGestureRecognizerStateChanged) {
      [selectionView setEvent : kButton1Motion atX : p.x andY : p.y];
      [selectionView setNeedsDisplay];
   } else if (panGesture.state == UIGestureRecognizerStateEnded) {
      [selectionView setEvent : kButton1Up atX : p.x andY : p.y];
      [selectionView setNeedsDisplay];
      selectionView.hidden = YES;
      [self setNeedsDisplay];
   }
}

//_________________________________________________________________
- (UIImage *) createImageForPicking
{
   assert(pad != nullptr && "initCGImageForPicking, pad is null");

   const CGRect rect = CGRectMake(0.f, 0.f, 640.f, 640.f);
   //Create bitmap context.
   UIGraphicsBeginImageContext(rect.size);
   CGContextRef ctx = UIGraphicsGetCurrentContext();

   //Now draw into this context.
   CGContextTranslateCTM(ctx, 0.f, rect.size.height);
   CGContextScaleCTM(ctx, 1.f, -1.f);
      
   //Disable anti-aliasing, to avoid "non-clear" colors.
   CGContextSetAllowsAntialiasing(ctx, 0);
   //Fill bitmap with black (nothing under cursor).
   CGContextSetRGBFillColor(ctx, 0.f, 0.f, 0.f, 1.f);
   CGContextFillRect(ctx, rect);
   //Set context and paint pad's contents
   //with special colors (color == object's identity)
   pad->SetContext(ctx);
   pad->PaintForSelection();
   
   UIImage * const uiImageForPicking = UIGraphicsGetImageFromCurrentImageContext();//autoreleased UIImage.
   UIGraphicsEndImageContext();
   
   return uiImageForPicking;
} 

//_________________________________________________________________
- (BOOL) fillPickingBufferFromImage : (UIImage *) image
{
   assert(image != nil && "fillPickingBufferFromImage:, parameter 'image' is nil");
   assert(pad != nullptr && "fillPickingBufferFromImage:, pad is null");

   CGImageRef cgImage = image.CGImage;

   const size_t pixelsW = CGImageGetWidth(cgImage);
   const size_t pixelsH = CGImageGetHeight(cgImage);
   //Declare the number of bytes per row. Each pixel in the bitmap
   //is represented by 4 bytes; 8 bits each of red, green, blue, and
   //alpha.
   const int bitmapBytesPerRow = pixelsW * 4;
   const int bitmapByteCount = bitmapBytesPerRow * pixelsH;
   
   //Use the generic RGB color space.
   CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
   if (!colorSpace) {
      //Log error: color space allocation failed.
      return NO;
   }

   try {
      std::vector<unsigned char> buffer(bitmapByteCount);
      CGContextRef ctx = CGBitmapContextCreate(&buffer[0], pixelsW, pixelsH, 8, bitmapBytesPerRow, colorSpace, kCGImageAlphaPremultipliedFirst);
      CGColorSpaceRelease(colorSpace);
      
      if (!ctx)
         return NO;

      const CGRect rect = CGRectMake(0.f, 0.f, pixelsW, pixelsH);
      //Draw the image to the bitmap context. Once we draw, the memory
      //allocated for the context for rendering will then contain the 
      //raw image data in the specified color space.
   
      CGContextSetAllowsAntialiasing(ctx, false);//Check, if I need this for a bitmap.
      CGContextDrawImage(ctx, rect, cgImage);

      pad->SetSelectionBuffer(pixelsW, pixelsH, &buffer[0]);
      // When finished, release the context
      CGContextRelease(ctx); 

      return YES;
   } catch (const std::bad_alloc &e) {
      CGColorSpaceRelease(colorSpace);
   }

   return NO;
}

//_________________________________________________________________
- (BOOL) initPadPicking
{
   UIImage * const pickImage = [self createImageForPicking];
   if (!pickImage)
      return NO;
   
   return [self fillPickingBufferFromImage : pickImage];
}

//_________________________________________________________________
- (void) handleTapGesture : (UITapGestureRecognizer *) tapGesture
{
   assert(tapGesture != nil && "handleTapGesture:, parameter 'tapGesture' is nil");

   if (processTap) {
      assert(pad != nullptr && "handleTapGesture:, pad is null");
   
      const CGPoint tapPt = [tapGesture locationInView : self];
      
      if (!pad->SelectionIsValid() && ![self initPadPicking])
         return;
      
      pad->Pick(tapPt.x, tapPt.y);
      
      if (pad->GetSelected()) {
         [selectionView setShowRotation : NO];
         [selectionView setPad : pad];
         [selectionView setNeedsDisplay];
         selectionView.hidden = NO;
      } else {
         selectionView.hidden = YES;
      }
   }
}

//_________________________________________________________________
- (void) setSelectionView : (SelectionView *) sv
{
   selectionView = sv;
}

//_________________________________________________________________
- (void) setProcessPan : (BOOL) p
{
   processPan = p;
}

//_________________________________________________________________
- (void) setProcessTap : (BOOL) t
{
   processTap = t;
}

@end
