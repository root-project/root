#import <CoreGraphics/CGContext.h>

#import "IOSSelectionMarkers.h"
#import "SelectionView.h"
#import "Constants.h"
#import "PadView.h"
#import "TAxis.h"

//C++ (ROOT) imports.
#import "IOSPad.h"

@implementation SelectionView {
   ROOT::iOS::Pad *pad;
}

@synthesize panActive;
@synthesize panStart;
@synthesize currentPanPoint;
@synthesize verticalPanDirection;

//____________________________________________________________________________________________________
+ (void) setShadowColor : (CGContextRef) ctx
{
   UIColor *shadowColor = [UIColor colorWithRed : 0.f green : 0.f blue : 0.f alpha : 0.7f];
   CGContextSetShadowWithColor(ctx, CGSizeMake(3.f, 3.f), 4.f, shadowColor.CGColor);
}

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame withPad : (ROOT::iOS::Pad *) p
{
   self = [super initWithFrame:frame];

   if (self) {
      pad = p;
      self.opaque = NO;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) showSelectedAxis : (CGContextRef)ctx withRect : (CGRect)rect
{
   //"Special case" function to show axis selection.
   using namespace ROOT::iOS;
   const CGFloat xMin = pad->GetUxmin();
   const CGFloat xMax = pad->GetUxmax();
   const CGFloat yMin = pad->GetUymin();
   const CGFloat yMax = pad->GetUymax();

   ROOT::iOS::SpaceConverter converter(rect.size.width, pad->GetX1(), pad->GetX2(), rect.size.height, pad->GetY1(), pad->GetY2());
   GraphicUtils::DrawSelectionMarker(ctx, CGPointMake(converter.XToView(xMin), converter.YToView(yMin)));

   const TAxis *axis = static_cast<TAxis *>(pad->GetSelected());
   if (!strcmp(axis->GetName(), "xaxis")) {
      GraphicUtils::DrawSelectionMarker(ctx, CGPointMake(converter.XToView(xMax), converter.YToView(yMin)));
   } else if (!strcmp(axis->GetName(), "yaxis")){
      GraphicUtils::DrawSelectionMarker(ctx, CGPointMake(converter.XToView(xMin), converter.YToView(yMax)));
   }//else is "Z" but we do not care.
}

//____________________________________________________________________________________________________
- (void)drawRect : (CGRect)rect
{
   CGContextRef ctx = UIGraphicsGetCurrentContext();

   CGContextSaveGState(ctx);

   CGContextClearRect(ctx, rect);
   CGContextTranslateCTM(ctx, 0.f, rect.size.height);
   CGContextScaleCTM(ctx, 1.f, -1.f);

   pad->cd();
   pad->SetViewWH(rect.size.width, rect.size.height);
   pad->SetContext(ctx);

   //Selected object will cast a shadow.
   //if (useShadows)
   if (!ROOT::iOS::Browser::deviceIsiPad3) {
      [SelectionView setShadowColor : ctx];
      pad->PaintSelected();
   } else {
      CGContextTranslateCTM(ctx, 2.5f, 2.5f);
      pad->PaintShadowForSelected();
      CGContextTranslateCTM(ctx, -2.5f, -2.5f);
      pad->PaintSelected();
   }

   //If we selected object has a polyline or polygon, markers will be painted.
   //But if it's a TAxis, I can not simply draw a marker for a line segment,
   //since TAxis has a lot of lines (ticks and marks).
   //I have to find the position for this markers here and paint them.
   if (dynamic_cast<TAxis *>(pad->GetSelected()))
      [self showSelectedAxis : ctx withRect : rect];

   CGContextRestoreGState(ctx);

   if (panActive) {
      CGContextSetRGBFillColor(ctx, 0.f, 0.f, 1.f, 0.2f);
      if (!verticalPanDirection)
         CGContextFillRect(ctx, CGRectMake(panStart.x, 0.f, currentPanPoint.x - panStart.x, rect.size.height));
      else
         CGContextFillRect(ctx, CGRectMake(0.f, panStart.y, rect.size.width, currentPanPoint.y - panStart.y));
   }
}

//____________________________________________________________________________________________________
- (BOOL)pointInside:(CGPoint)point withEvent:(UIEvent *) event
{
   //Thanks to gyim,
   //http://stackoverflow.com/questions/1694529/allowing-interaction-with-a-uiview-under-another-uiview
   return NO;
}

//____________________________________________________________________________________________________
- (void) setPad : (ROOT::iOS::Pad *)p
{
   pad = p;
}

@end
