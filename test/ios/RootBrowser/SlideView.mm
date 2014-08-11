#import <QuartzCore/QuartzCore.h>
#import <CoreGraphics/CGContext.h>

#import "SlideView.h"

//C++ (ROOT) imports.
#import "IOSPad.h"

const CGRect slideFrame = CGRectMake(0.f, 0.f, 650.f, 650.f);

@implementation SlideView {
   ROOT::iOS::Pad *pad;
}

//____________________________________________________________________________________________________
+ (CGSize) slideSize
{
   return slideFrame.size;
}

//____________________________________________________________________________________________________
+ (CGRect) slideFrame
{
   return slideFrame;
}

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame
{
   self = [super initWithFrame : frame];

   if (self) {
      self.layer.shadowOpacity = 0.3f;
      self.layer.shadowColor = [UIColor blackColor].CGColor;
      self.layer.shadowOffset = CGSizeMake(10.f, 10.f);
      self.layer.shadowPath = [UIBezierPath bezierPathWithRect : slideFrame].CGPath;
   }

   return self;
}

- (void) setPad : (ROOT::iOS::Pad *)newPad
{
   pad = newPad;
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect)rect
{
   CGContextRef ctx = UIGraphicsGetCurrentContext();

   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   CGContextFillRect(ctx, rect);

   if (!pad)
      return;

   CGContextTranslateCTM(ctx, 0.f, rect.size.height);
   CGContextScaleCTM(ctx, 1.f, -1.f);

   pad->cd();
   pad->SetViewWH(rect.size.width, rect.size.height);
   pad->SetContext(ctx);
   pad->Paint();
}


@end
