#import <stddef.h>
#import <stdlib.h>
#import <math.h>

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

- (void) handlePanGesture : (UIPanGestureRecognizer *)panGesture;
- (void) handleTapGesture : (UITapGestureRecognizer *)tapGesture;

@end

@implementation PadView

//_________________________________________________________________
- (id) initWithFrame:(CGRect)frame forPad : (ROOT::iOS::Pad*)pd
{
   self = [super initWithFrame : frame];

   if (self) {
      //Initialize C++ objects here.
      pad = pd;

      scaleFactor = frame.size.width / 640.f;
   }

   return self;
}

//_________________________________________________________________
- (void)drawRect : (CGRect)rect
{
   // Drawing code
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
   pad->Clear();
}

//_________________________________________________________________
- (void) handlePanGesture : (UIPanGestureRecognizer *)panGesture
{
   if (!processPan)
      return;

   const CGPoint p = [panGesture locationInView:self];
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
- (CGImageRef) initCGImageForPicking
{
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
   
   UIImage *uiImageForPicking = UIGraphicsGetImageFromCurrentImageContext();//autoreleased UIImage.
   CGImageRef cgImageForPicking = uiImageForPicking.CGImage;
   CGImageRetain(cgImageForPicking);//It must live as long, as I need :)
   
   UIGraphicsEndImageContext();
   
   return cgImageForPicking;

} 

//_________________________________________________________________
- (BOOL) fillPickingBufferFromCGImage : (CGImageRef) cgImage
{
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
	
   unsigned char *buffer = (unsigned char*)malloc(bitmapByteCount);
   if (!buffer) {
      //Log error: memory allocation failed.
      CGColorSpaceRelease(colorSpace);
      return NO;
   }

	// Create the bitmap context. We want pre-multiplied ARGB, 8-bits 
	// per component. Regardless of what the source image format is 
	// (CMYK, Grayscale, and so on) it will be converted over to the format
	// specified here by CGBitmapContextCreate.
   CGContextRef ctx = CGBitmapContextCreate(buffer, pixelsW, pixelsH, 8, bitmapBytesPerRow, colorSpace, kCGImageAlphaPremultipliedFirst);

   CGColorSpaceRelease(colorSpace);

	if (!ctx) {
      //Log error: bitmap context creation failed.
      free(buffer);
      return NO;
   }
	
	const CGRect rect = CGRectMake(0.f, 0.f, pixelsW, pixelsH); 
	//Draw the image to the bitmap context. Once we draw, the memory 
	//allocated for the context for rendering will then contain the 
	//raw image data in the specified color space.
   
   CGContextSetAllowsAntialiasing(ctx, 0);//Check, if I need this for a bitmap.
	CGContextDrawImage(ctx, rect, cgImage);

   pad->SetSelectionBuffer(pixelsW, pixelsH, buffer);
	// When finished, release the context
	CGContextRelease(ctx); 
   free(buffer);

   return YES;
}

//_________________________________________________________________
- (BOOL) initPadPicking
{
   CGImageRef cgImage = [self initCGImageForPicking];
   if (!cgImage)
      return NO;

   const BOOL res = [self fillPickingBufferFromCGImage : cgImage];
   CGImageRelease(cgImage);
   
   return res;
}

//_________________________________________________________________
- (void) handleTapGesture : (UITapGestureRecognizer *) tapGesture
{
   if (processTap) {
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
- (void) setSelectionView:(SelectionView *)sv
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
