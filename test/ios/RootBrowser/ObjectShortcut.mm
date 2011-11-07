#import <CoreGraphics/CGContext.h>
#import <QuartzCore/QuartzCore.h>

#import "FileContentController.h"
#import "ObjectShortcut.h"

//C++ (ROOT) imports.
#import "IOSFileContainer.h"
#import "TObject.h"

@implementation ObjectShortcut  {
   __weak FileContentController *controller;

   unsigned objectIndex;
   NSString *objectName;
}

@synthesize icon;
@synthesize objectIndex;
@synthesize objectName;

//____________________________________________________________________________________________________
+ (CGFloat) iconWidth
{
   return 150.f;
}

//____________________________________________________________________________________________________
+ (CGFloat) iconHeight
{
   return 150.f;
}

//____________________________________________________________________________________________________
+ (CGFloat) textHeight
{
   return 100.f;
}

//____________________________________________________________________________________________________
+ (CGRect) defaultRect
{
   return CGRectMake(0.f, 0.f, [ObjectShortcut iconWidth], [ObjectShortcut iconHeight] + [ObjectShortcut textHeight]);
}

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame controller : (FileContentController*) c forObjectAtIndex : (unsigned)objIndex withThumbnail : (UIImage *)thumbnail;
{
   self = [super initWithFrame:frame];

   if (self) {   
      //ROOT's staff.
      controller = c;
      objectIndex = objIndex;
      
      const TObject *obj = controller.fileContainer->GetObject(objIndex);
      self.objectName = [NSString stringWithFormat : @"%s", obj->GetName()];
      
      self.icon = thumbnail;
   
      //Geometry and a shadow.
      self.layer.shadowColor = [UIColor blackColor].CGColor;
      self.layer.shadowOpacity = 0.3;
      self.layer.shadowOffset = CGSizeMake(10.f, 10.f);
      frame.origin = CGPointZero;
      frame.size.height = [ObjectShortcut iconHeight];
      self.layer.shadowPath = [UIBezierPath bezierPathWithRect : frame].CGPath;
      
      self.opaque = NO;
      
      //Tap gesture to select an object.
      UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(handleTap)];
      [self addGestureRecognizer : tap];
   }

   return self;
}

//____________________________________________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   [icon drawAtPoint:CGPointZero];
   
   CGContextRef ctx = UIGraphicsGetCurrentContext();
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   const CGRect textRect = CGRectMake(0.f, [ObjectShortcut iconHeight], [ObjectShortcut iconWidth], [ObjectShortcut textHeight]);
   [objectName drawInRect : textRect withFont : [UIFont systemFontOfSize : 16] lineBreakMode : UILineBreakModeWordWrap alignment : UITextAlignmentCenter];
}

//____________________________________________________________________________________________________
- (void) handleTap
{
   [controller selectObjectFromFile : self];
}

@end
