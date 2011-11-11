#import <stddef.h>
#import <string.h>
#import <stdlib.h>
#import <math.h>

#import <CoreGraphics/CoreGraphics.h>
#import <CoreGraphics/CGContext.h>

#import "ROOTObjectController.h"
#import "SelectionView.h"
#import "Constants.h"
#import "PadView.h"

#import "TAxis.h"

//C++ code (ROOT's ios module)
#import "IOSPad.h"

const CGFloat tapInterval = 0.15f;

@interface PadView () {
   ROOT::iOS::Pad *pad;
   
   __weak ROOTObjectController *controller;
   
   CGFloat currentScale;

   BOOL panActive;
   
   CGPoint tapPt;
   BOOL processSecondTap;
}

- (void) handleSingleTap;
- (void) handleDoubleTap;

@end

@implementation PadView

@synthesize selectionView;

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame controller : (ROOTObjectController *)c forPad : (ROOT::iOS::Pad*)pd
{
   self = [super initWithFrame : frame];

   if (self) {
      controller = c;
      pad = pd;
      
      frame.origin = CGPointZero;
      selectionView = [[SelectionView alloc] initWithFrame : frame withPad : pad];
      selectionView.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
      selectionView.hidden = YES;
      [self addSubview : selectionView];
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) setPad : (ROOT::iOS::Pad *)newPad
{
   pad = newPad;
   [selectionView setPad : newPad];
}

//____________________________________________________________________________________________________
- (void)drawRect : (CGRect)rect
{
   // Drawing code   
   
   CGContextRef ctx = UIGraphicsGetCurrentContext();
   CGContextClearRect(ctx, rect);

   pad->SetViewWH(rect.size.width, rect.size.height);
   CGContextTranslateCTM(ctx, 0.f, rect.size.height);
   CGContextScaleCTM(ctx, 1.f, -1.f);
   pad->cd();
   pad->SetContext(ctx);
   pad->Paint();
   
   if (!selectionView.hidden)
      [selectionView setNeedsDisplay];
}

//____________________________________________________________________________________________________
- (void) clearPad
{
   pad->Clear();
}

//____________________________________________________________________________________________________
- (void) addPanRecognizer
{
   panActive = YES;
}

//____________________________________________________________________________________________________
- (void) removePanRecognizer
{
   panActive = NO;
}

#pragma mark - Picking related stuff here.

//____________________________________________________________________________________________________
- (CGImageRef) initCGImageForPicking
{
   using namespace ROOT::iOS::Browser;
   const CGRect rect = CGRectMake(0.f, 0.f, padW, padH);
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
   pad->SetViewWH(rect.size.width, rect.size.height);

   pad->cd();

   pad->SetContext(ctx);
   pad->PaintForSelection();
   
   UIImage *uiImageForPicking = UIGraphicsGetImageFromCurrentImageContext();//autoreleased UIImage.
   CGImageRef cgImageForPicking = uiImageForPicking.CGImage;
   CGImageRetain(cgImageForPicking);//It must live as long, as I need :)
   
   UIGraphicsEndImageContext();
   
   return cgImageForPicking;

} 

//____________________________________________________________________________________________________
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

//____________________________________________________________________________________________________
- (BOOL) initPadPicking
{
   CGImageRef cgImage = [self initCGImageForPicking];
   if (!cgImage)
      return NO;

   const BOOL res = [self fillPickingBufferFromCGImage : cgImage];
   CGImageRelease(cgImage);
   
   return res;
}

//____________________________________________________________________________________________________
- (CGPoint) scaledPoint : (CGPoint)pt
{
   const CGFloat scale = ROOT::iOS::Browser::padW / self.frame.size.width;
   return CGPointMake(pt.x * scale, pt.y * scale);
}

//____________________________________________________________________________________________________
- (BOOL) pointOnSelectedObject : (CGPoint) pt
{
   //check if there is any object in pt.

   if (!pad->SelectionIsValid() && ![self initPadPicking])
      return NO;

   const CGPoint newPt = [self scaledPoint : pt];
   if (pad->GetSelected() == pad->ObjectInPoint(newPt.x, newPt.y))
      return YES;

   return NO;
}

//____________________________________________________________________________________________________
- (void) handleSingleTap
{
   //Make a selection, fill the editor, disable double tap.
   const CGPoint scaledTapPt = [self scaledPoint : tapPt];
   if (!pad->SelectionIsValid() && ![self initPadPicking])
      return;
      
   pad->Pick(scaledTapPt.x, scaledTapPt.y);
   //Tell controller that selection has probably changed.
   [controller objectWasSelected : pad->GetSelected()];
   processSecondTap = NO;
}

//____________________________________________________________________________________________________
- (void) touchesBegan : (NSSet *)touches withEvent : (UIEvent *)event
{
   UITouch *touch = [touches anyObject];
   if (touch.tapCount == 1) {
      //Interaction has started.
      tapPt = [touch locationInView : self];
      //Gesture can be any of them:
      processSecondTap = YES;
   } 
}

//____________________________________________________________________________________________________
- (void) touchesMoved : (NSSet *)touches withEvent : (UIEvent *)event
{
   if (panActive) {
      processSecondTap = NO;
      TObject *selected = pad->GetSelected();
      if (TAxis *axis = dynamic_cast<TAxis *>(selected)) {
         if (!selectionView.panActive) {
            selectionView.panActive = YES;
            if (!strcmp(axis->GetName(), "xaxis"))
               selectionView.verticalDirection = NO;
            else
               selectionView.verticalDirection = YES;
            selectionView.panStart = tapPt;
            
            pad->ExecuteEventAxis(kButton1Down, tapPt.x, tapPt.y, axis);
         } else {
            const CGPoint newPt = [[touches anyObject] locationInView : self];
            selectionView.currentPanPoint = newPt;
            pad->ExecuteEventAxis(kButton1Motion, newPt.x, newPt.y, axis);
            [selectionView setNeedsDisplay];
         }
      } else {
         //We move object in a canvas now.
      }
   }
}

//____________________________________________________________________________________________________
- (void) touchesEnded : (NSSet *)touches withEvent : (UIEvent *)event
{
   UITouch *touch = [touches anyObject];
   if (touch.tapCount == 1 && !panActive) {
      [self performSelector : @selector(handleSingleTap) withObject : nil afterDelay : tapInterval];
   } else if (touch.tapCount == 2 && processSecondTap) {
      [NSObject cancelPreviousPerformRequestsWithTarget : self];
      [self handleDoubleTap];
   }
   
   if (selectionView.panActive) {
      panActive = NO;
      selectionView.panActive = NO;
      const CGPoint pt = [touch locationInView : self];
      pad->ExecuteEventAxis(kButton1Up, pt.x, pt.y, (TAxis *)pad->GetSelected());
      pad->InvalidateSelection(kTRUE);
      [self setNeedsDisplay];

      UIScrollView *parent = (UIScrollView *)[self superview];
      parent.canCancelContentTouches = YES;
      parent.delaysContentTouches = YES;
   }
}

//____________________________________________________________________________________________________
- (void) handleDoubleTap
{
   //This is zoom/unzoom action or axis unzoom.
   const CGPoint scaledTapPt = [self scaledPoint : tapPt];
   TAxis *axis = dynamic_cast<TAxis *>(pad->GetSelected());

   if (!pad->SelectionIsValid() && ![self initPadPicking])
      return;

   if (axis && pad->ObjectInPoint(scaledTapPt.x, scaledTapPt.y) == axis) {
      axis->UnZoom();
      pad->InvalidateSelection(kTRUE);
      [self setNeedsDisplay];
   } else {
      [controller handleDoubleTapOnPad : tapPt];
   }
   
   UIScrollView *parent = (UIScrollView *)[self superview];
   parent.canCancelContentTouches = YES;
   parent.delaysContentTouches = YES;
}

@end
