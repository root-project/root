#import "SelectionView.h"

//C++ (ROOT)
#import "IOSPad.h"


@implementation SelectionView

//______________________________________________________________________________
- (id)initWithFrame:(CGRect)frame
{
    self = [super initWithFrame : frame];
    if (self) {
        // Initialization code
        self.opaque = NO;
    }

    return self;
}

//______________________________________________________________________________
- (void)dealloc
{
    [super dealloc];
}

//______________________________________________________________________________
- (void) setPad : (ROOT::iOS::Pad *)newPad
{
   pad = newPad;
}

//______________________________________________________________________________
- (void) setEvent : (int) e atX : (int) x andY : (int) y
{
   ev = e;
   px = x;
   py = y;
}

//______________________________________________________________________________
- (void) drawRect:(CGRect)rect
{
   if (!pad)
      return;

   CGContextRef ctx = UIGraphicsGetCurrentContext();
   CGContextClearRect(ctx, rect);

   CGContextTranslateCTM(ctx, 0.f, rect.size.height);
   CGContextScaleCTM(ctx, 1.f, -1.f);
   
   pad->cd();
   pad->SetContext(ctx);
   if (showRotation) {
      pad->ExecuteRotateView(ev, px, py);
   } else {
      CGContextTranslateCTM(ctx, 2.5f, 2.5f);
      pad->PaintShadowForSelected();
      CGContextTranslateCTM(ctx, -2.5f, -2.5f);
      pad->PaintSelected();
   }
}

//______________________________________________________________________________
- (void) setShowRotation : (BOOL) show
{
   showRotation = show;
}

//______________________________________________________________________________
- (BOOL)pointInside:(CGPoint)point withEvent:(UIEvent *) event 
{
   //Thanks to gyim, 
   //http://stackoverflow.com/questions/1694529/allowing-interaction-with-a-uiview-under-another-uiview
   return NO;
}

@end
