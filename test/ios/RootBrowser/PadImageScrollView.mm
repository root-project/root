#import <stdlib.h>

#import "PadImageScrollView.h"
#import "PadImageView.h"

//C++ (ROOT) imports.
#import "TObject.h"
#import "IOSPad.h"

static const CGFloat defaultImageW = 700.f;
static const CGFloat defaultImageH = 700.f;
static const CGFloat maxZoom = 2.f;
static const CGFloat minZoom = 1.f;

@implementation PadImageScrollView {
   ROOT::iOS::Pad *pad;
   
   PadImageView *nestedView;
}

@synthesize padImage;

//____________________________________________________________________________________________________
+ (CGRect) defaultImageFrame
{
   return CGRectMake(0.f, 0.f, 700.f, 700.f);
}

//____________________________________________________________________________________________________
- (CGPoint) adjustOriginForFrame : (CGRect)frame withSize : (CGSize) sz
{
   return CGPointMake(frame.size.width / 2 - sz.width / 2, frame.size.height / 2 - sz.height / 2);
}

//____________________________________________________________________________________________________
- (void) initPadImageView : (CGRect)frame
{
   CGRect padFrame = [PadImageScrollView defaultImageFrame];
   padFrame.origin = [self adjustOriginForFrame : frame withSize : padFrame.size];
   
   nestedView = [[PadImageView alloc] initWithFrame : padFrame];
   nestedView.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   
   [self addSubview : nestedView];
}

//____________________________________________________________________________________________________
- (void) setContentSize : (CGSize) size contentOffset : (CGPoint)offset minScale : (CGFloat)min maxScale : (CGFloat)max scale : (CGFloat)scale
{
   self.maximumZoomScale = max;
   self.minimumZoomScale = min;
   self.zoomScale = scale;
   self.contentSize = size;
   self.contentOffset = offset;
}

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame
{
   if (self = [super initWithFrame : frame]) {
      self.delegate = self; //hehehehe
      self.bouncesZoom = NO;
      self.bounces = NO;
      self.backgroundColor = [UIColor clearColor];
      self.decelerationRate = UIScrollViewDecelerationRateFast;

      [self setContentSize : frame.size contentOffset : CGPointZero minScale : minZoom maxScale : maxZoom scale : 1];
      
      UITapGestureRecognizer *doubleTap = [[UITapGestureRecognizer alloc] initWithTarget : self action : @selector(handleDoubleTap:)];
      doubleTap.numberOfTapsRequired = 2;
      [self addGestureRecognizer : doubleTap];
   }
    
   return self;
}

#pragma mark - Child view's management
//____________________________________________________________________________________________________
- (void) clearScroll
{
   [self setContentSize : [PadImageScrollView defaultImageFrame].size contentOffset : CGPointZero minScale : minZoom maxScale : maxZoom scale : 1];
   [nestedView removeFromSuperview];
   nestedView = nil;
}

#pragma mark - Image/pad/geometry management.
//____________________________________________________________________________________________________
- (UIImage *) createPadImage : (CGSize) imageSize
{
   UIGraphicsBeginImageContext(imageSize);
   CGContextRef ctx = UIGraphicsGetCurrentContext();

   if (!ctx) {
      NSLog(@"UIGraphicsGetCurrentContext failed");
      UIGraphicsEndImageContext();
      return nil;
   }

   //Now draw into this context.
   CGContextTranslateCTM(ctx, 0.f, imageSize.height);
   CGContextScaleCTM(ctx, 1.f, -1.f);
      
   const CGRect imageFrame = CGRectMake(0.f, 0.f, imageSize.width, imageSize.height);
   //Fill bitmap with white first.
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   CGContextFillRect(ctx, imageFrame);
   //
   pad->cd();
   pad->SetContext(ctx);
   pad->SetViewWH(imageSize.width, imageSize.height);
   pad->Paint();

   //Ready.
   UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
   UIGraphicsEndImageContext();
       
   return image;
}

//____________________________________________________________________________________________________
- (void) setPad : (ROOT::iOS::Pad *)p
{
   pad = p;
   //Generate new image.
   padImage = [self createPadImage : [PadImageScrollView defaultImageFrame].size];
   
   pad->SetViewWH(defaultImageW, defaultImageH);

   if (nestedView && nestedView.zoomed) {
      [self clearScroll];
      [self initPadImageView : self.frame];
   } else if (!nestedView) {
      [self initPadImageView : self.frame];
   }

   nestedView.padImage = padImage;
   [nestedView setNeedsDisplay];
}

//____________________________________________________________________________________________________
- (void) setPad : (ROOT::iOS::Pad *)p andImage : (UIImage *)image
{
   pad = p;
   padImage = image;
   
   if (nestedView && nestedView.zoomed) {
      [self clearScroll];
      [self initPadImageView : self.frame];
      
   } else if (!nestedView) {
      [self initPadImageView : self.frame];
   }
   
   nestedView.padImage = padImage;
   [nestedView setNeedsDisplay];
}

//____________________________________________________________________________________________________
- (void) resetToFrame : (CGRect) newFrame
{
   self.frame = newFrame; 
   [self setContentSize : newFrame.size contentOffset : CGPointZero minScale : minZoom maxScale : maxZoom scale : 1];
   
   if (nestedView.zoomed) {
      [self clearScroll];
      [self initPadImageView : newFrame];
      nestedView.padImage = padImage;
      [nestedView setNeedsDisplay];
   } else {
      CGRect padFrame = [PadImageScrollView defaultImageFrame];
      padFrame.origin = [self adjustOriginForFrame : newFrame withSize : padFrame.size];
   }
}

//
//_________________________________________________________________
- (CGRect)centeredFrameForScrollView:(UIScrollView *)scroll andUIView:(UIView *)rView 
{
   CGSize boundsSize = scroll.bounds.size;
   CGRect frameToCenter = rView.frame;
   // center horizontally
   if (frameToCenter.size.width < boundsSize.width) {
      frameToCenter.origin.x = (boundsSize.width - frameToCenter.size.width) / 2;
   }
   else {
      frameToCenter.origin.x = 0;
   }
   // center vertically
   if (frameToCenter.size.height < boundsSize.height) {
      frameToCenter.origin.y = (boundsSize.height - frameToCenter.size.height) / 2;
   }
   else {
      frameToCenter.origin.y = 0;
   }
   
   return frameToCenter;
}

//____________________________________________________________________________________________________
- (void)scrollViewDidZoom:(UIScrollView *)scroll
{
   nestedView.frame = [self centeredFrameForScrollView : scroll andUIView : nestedView];
}

//____________________________________________________________________________________________________
- (void)scrollViewDidEndZooming : (UIScrollView *)scroll withView : (UIView *)view atScale : (float)scale
{
   const CGPoint offset = [scroll contentOffset];
   const CGRect newFrame = nestedView.frame;
  
   [scroll setZoomScale : 1.f];
   
   const unsigned base = [PadImageScrollView defaultImageFrame].size.width;

   scroll.minimumZoomScale = base / newFrame.size.width;
   scroll.maximumZoomScale = maxZoom * base / newFrame.size.width;

   [nestedView removeFromSuperview];

   nestedView = [[PadImageView alloc] initWithFrame : newFrame];

   UIImage *image = [self createPadImage : newFrame.size];
   nestedView.padImage = image;
   [scroll addSubview : nestedView];

   scroll.contentSize = newFrame.size;
   scroll.contentOffset = offset;

   nestedView.zoomed = YES;
}

//____________________________________________________________________________________________________
- (UIView *) viewForZoomingInScrollView:(UIScrollView *)scrollView
{
   return nestedView;
}

//____________________________________________________________________________________________________
- (CGRect)zoomRectForScale:(float)scale withCenter:(CGPoint)center {
    
    CGRect zoomRect;
    
    // the zoom rect is in the content view's coordinates. 
    //    At a zoom scale of 1.0, it would be the size of the imageScrollView's bounds.
    //    As the zoom scale decreases, so more content is visible, the size of the rect grows.
    zoomRect.size.height = [self frame].size.height / scale;
    zoomRect.size.width  = [self frame].size.width  / scale;
    
    // choose an origin so as to get the right center.
    zoomRect.origin.x    = center.x - (zoomRect.size.width  / 2.0);
    zoomRect.origin.y    = center.y - (zoomRect.size.height / 2.0);
    
    return zoomRect;
}

//____________________________________________________________________________________________________
- (void) handleDoubleTap : (UITapGestureRecognizer *)tap
{
   //Identify, if we should unzoom.
   if (fabs(nestedView.frame.size.width - maxZoom * defaultImageW) < 10) {
      [self resetToFrame : self.frame];
   } else {
      //Zoom in.
      const CGFloat newScale = maxZoom * defaultImageW / nestedView.frame.size.width;
      CGRect zoomRect = [self zoomRectForScale : newScale withCenter : [tap locationInView : self]];
      [self zoomToRect : zoomRect animated : YES];
   }
}

@end
