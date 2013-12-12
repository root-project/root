#import <cstddef>
#import <cstring>
#import <vector>

#import "ObjectViewController.h"
#import "PadSelectionView.h"
#import "Constants.h"
#import "PadView.h"

#import "TAxis.h"

//C++ code (ROOT's ios module)
#import "IOSPad.h"

const CGFloat tapInterval = 0.15f;

@implementation PadView {
   __weak ObjectViewController *controller;

   ROOT::iOS::Pad *pad;

   BOOL panActive;
   
   CGPoint tapPt;
   BOOL processSecondTap;
   
   BOOL isMutable;
}

@synthesize selectionView;
@synthesize zoomed;

//____________________________________________________________________________________________________
- (instancetype) initWithFrame : (CGRect) frame controller : (ObjectViewController *) c forPad : (ROOT::iOS::Pad*) pd
{
   assert(c != nil && "initWithFrame:forPad:, parameter 'c' is nil");
   assert(pd != nullptr && "initWithFrame:forPad:, parameter 'pd' is null");

   if (self = [super initWithFrame : frame]) {
      controller = c;
      pad = pd;
      
      isMutable = YES;
      
      frame.origin = CGPointZero;
      selectionView = [[PadSelectionView alloc] initWithFrame : frame withPad : pad];
      selectionView.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin |
                                       UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
      selectionView.hidden = YES;
      [self addSubview : selectionView];
   }

   return self;
}

//____________________________________________________________________________________________________
- (instancetype) initImmutableViewWithFrame : (CGRect) frame
{
   if (self = [super initWithFrame : frame]) {
      controller = nil;
      pad = nullptr;
      selectionView = nil;

      isMutable = NO;
      
      self.multipleTouchEnabled = NO;
   }
   
   return self;
}

//____________________________________________________________________________________________________
- (void) setPad : (ROOT::iOS::Pad *) newPad
{
   assert(newPad != nullptr && "setPad:, parameter 'newPad' is null");

   pad = newPad;
   if (isMutable)
      [selectionView setPad : newPad];
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect) rect
{
   // Drawing code
   if (!pad)//assert instead???
      return;

   CGContextRef ctx = UIGraphicsGetCurrentContext();
   CGContextClearRect(ctx, rect);

   pad->SetViewWH(rect.size.width, rect.size.height);
   CGContextTranslateCTM(ctx, 0.f, rect.size.height);
   CGContextScaleCTM(ctx, 1.f, -1.f);

   pad->cd();
   pad->SetContext(ctx);
   pad->Paint();
   
   if (isMutable && !selectionView.hidden)
      [selectionView setNeedsDisplay];
}

//____________________________________________________________________________________________________
- (void) clearPad
{
   assert(pad != nullptr && "clearPad, pad is null");
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

#pragma mark - Picking/gesture handling and related stuff here.

//____________________________________________________________________________________________________
- (UIImage *) createImageForPicking
{
   assert(pad != nullptr && "createImageForPicking, pad is null");

   if (!isMutable)
      return nil;

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
   
   UIImage * const uiImageForPicking = UIGraphicsGetImageFromCurrentImageContext();//autoreleased UIImage.
   
   UIGraphicsEndImageContext();
   
   return uiImageForPicking;
} 

//____________________________________________________________________________________________________
- (BOOL) fillPickingBufferFromImage : (UIImage *) image
{
   assert(image != nil && "fillPickingBufferFromImage:, parameter 'image' is nil");
   assert(pad != nullptr && "fillPickingBufferFromImage:, pad is null");

   if (!isMutable)
      return NO;

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
      // Create the bitmap context. We want pre-multiplied ARGB, 8-bits
      // per component. Regardless of what the source image format is 
      // (CMYK, Grayscale, and so on) it will be converted over to the format
      // specified here by CGBitmapContextCreate.
      CGContextRef ctx = CGBitmapContextCreate(&buffer[0], pixelsW, pixelsH, 8, bitmapBytesPerRow, colorSpace, kCGImageAlphaPremultipliedFirst);
      CGColorSpaceRelease(colorSpace);

      if (!ctx)
         return NO;

      const CGRect rect = CGRectMake(0.f, 0.f, pixelsW, pixelsH);
      //Draw the image to the bitmap context. Once we draw, the memory 
      //allocated for the context for rendering will then contain the 
      //raw image data in the specified color space.
      
      CGContextSetAllowsAntialiasing(ctx, 0);//Check, if I need this for a bitmap.
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

//____________________________________________________________________________________________________
- (BOOL) initPadPicking
{
   if (!isMutable)
      return NO;

   UIImage * const image = [self createImageForPicking];
   if (!image)
      return NO;

   return [self fillPickingBufferFromImage : image];
}

//____________________________________________________________________________________________________
- (CGPoint) scaledPoint : (CGPoint) pt
{
   const CGFloat scale = ROOT::iOS::Browser::padW / self.frame.size.width;
   return CGPointMake(pt.x * scale, pt.y * scale);
}

//____________________________________________________________________________________________________
- (BOOL) pointOnSelectedObject : (CGPoint) pt
{
   //check if there is any object in pt.
   assert(pad != nullptr && "pointOnSelectedObject:, pad is null");
   
   if (!isMutable)
      return NO;

   if (!pad->SelectionIsValid() && ![self initPadPicking])
      return NO;

   const CGPoint newPt = [self scaledPoint : pt];
   return pad->GetSelected() == pad->ObjectInPoint(newPt.x, newPt.y);
}

//Touch events processing.

//____________________________________________________________________________________________________
- (void) handleSingleTap
{
   //Make a selection, fill the editor, disable double tap.
   assert(pad != nullptr && "handleSingleTap, pad is null");
   
   if (!isMutable)
      return;

   if (!pad->SelectionIsValid() && ![self initPadPicking])
      return;

   const CGPoint scaledTapPt = [self scaledPoint : tapPt];
   pad->Pick(scaledTapPt.x, scaledTapPt.y);
   //Tell controller that selection has probably changed.
   [controller objectWasSelected : pad->GetSelected()];
   
   //There were no second tap withing tapInterval, ignore the subsequent tap.
   processSecondTap = NO;
}

//____________________________________________________________________________________________________
- (void) touchesBegan : (NSSet *) touches withEvent : (UIEvent *) event
{
#pragma unused(event)

   assert(touches != nil && "touchesBegan:withEvent:, parameter 'touches' is nil");

   if (!isMutable)
      return;

   UITouch * const touch = [touches anyObject];
   if (touch.tapCount == 1) {
      //The first tap - interaction has started.
      tapPt = [touch locationInView : self];
      //Gesture can be any of them:
      processSecondTap = YES;
   } 
}

//____________________________________________________________________________________________________
- (void) touchesMoved : (NSSet *) touches withEvent : (UIEvent *) event
{
#pragma unused(event)

   assert(touches != nil && "touchesMoved:withEvent:, parameter 'touches' is nil");
   assert(pad != nullptr && "touchesMoved:withEvent:, pad is null");

   if (!isMutable)
      return;

   if (panActive) {
      processSecondTap = NO;
      TObject *selected = pad->GetSelected();
      if (TAxis *axis = dynamic_cast<TAxis *>(selected)) {
         if (!selectionView.panActive) {
            selectionView.panActive = YES;
            if (!std::strcmp(axis->GetName(), "xaxis"))
               selectionView.verticalPanDirection = NO;
            else
               selectionView.verticalPanDirection = YES;
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
- (void) touchesEnded : (NSSet *) touches withEvent : (UIEvent *) event
{
#pragma unused(event)

   assert(touches != nil && "touchesEnded:withEvent:, parameter 'touches' is nil");
   assert(pad != nullptr && "touchesEnded:withEvent:, pad is null");

   if (!isMutable)
      return;

   UITouch * const touch = [touches anyObject];
   if (touch.tapCount == 1 && !panActive) {
      [self performSelector : @selector(handleSingleTap) withObject : nil afterDelay : tapInterval];
   } else if (touch.tapCount == 2 && processSecondTap) {
      //The second tap was done withing tapInterval, thus we
      //should process the gesture as a double tap.
      //Cancel handleSingleTap:
      [NSObject cancelPreviousPerformRequestsWithTarget : self];
      //
      [self handleDoubleTap];
   }

   if (selectionView.panActive) {
      panActive = NO;
      selectionView.panActive = NO;
      const CGPoint pt = [touch locationInView : self];
      pad->ExecuteEventAxis(kButton1Up, pt.x, pt.y, (TAxis *)pad->GetSelected());
      pad->InvalidateSelection(kTRUE);
      [self setNeedsDisplay];

      UIScrollView * const parent = (UIScrollView *)[self superview];
      parent.canCancelContentTouches = YES;
      parent.delaysContentTouches = YES;
   }
}

//____________________________________________________________________________________________________
- (void) handleDoubleTap
{
   //This is zoom/unzoom action or axis unzoom.
   
   assert(pad != nullptr && "handleDoubleTap, pad is null");
   
   if (!isMutable)
      return;
   
   const CGPoint scaledTapPt = [self scaledPoint : tapPt];
   TAxis * const axis = dynamic_cast<TAxis *>(pad->GetSelected());

   if (!pad->SelectionIsValid() && ![self initPadPicking])
      return;

   if (axis && pad->ObjectInPoint(scaledTapPt.x, scaledTapPt.y) == axis) {
      axis->UnZoom();
      pad->InvalidateSelection(kTRUE);
      [self setNeedsDisplay];
   } else {
      [controller handleDoubleTapOnPad : tapPt];
   }
   
   UIScrollView * const parent = (UIScrollView *)[self superview];
   parent.canCancelContentTouches = YES;
   parent.delaysContentTouches = YES;
}

@end
