#import <CoreGraphics/CGContext.h>
#import <QuartzCore/QuartzCore.h>
#import <Availability.h>

#import "FileContentController.h"
#import "ObjectShortcut.h"
#import "SpotView.h"

//C++ (ROOT) imports.
#import "FileUtils.h"
#import "TObject.h"

const CGSize folderIconSize = CGSizeMake(128.f, 128.f);

@implementation ObjectShortcut  {
   __weak FileContentController *controller;

   unsigned objectIndex;
   NSString *objectName;
}

@synthesize isDirectory;
@synthesize icon;
@synthesize objectIndex;
@synthesize objectName;
@synthesize spot;

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
- (id) initWithFrame : (CGRect)frame controller : (FileContentController*) c forFolderAtIndex : (unsigned)index
{
   using namespace ROOT::iOS::Browser;

   if (self = [super initWithFrame : frame]) {
      frame.origin = CGPointZero;
      frame.size.height = [ObjectShortcut iconHeight];

      spot = [[SpotView alloc] initWithFrame : frame];
      [self addSubview : spot];

      controller = c;
      objectIndex = index;

      const FileContainer *cont = controller.fileContainer->GetDirectory(index);
      isDirectory = YES;
      self.objectName = [NSString stringWithFormat : @"%s", cont->GetFileName()];
      self.icon = [UIImage imageNamed : @"directory.png"];
      self.opaque = NO;

      //Tap gesture to select a directory.
      UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(handleTap)];
      [self addGestureRecognizer : tap];
   }

   return self;
}

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame controller : (FileContentController*) c forObjectAtIndex : (unsigned)objIndex withThumbnail : (UIImage *)thumbnail
{
   using namespace ROOT::iOS::Browser;

   self = [super initWithFrame:frame];

   if (self) {
      frame.origin = CGPointZero;
      frame.size.height = [ObjectShortcut iconHeight];

      spot = [[SpotView alloc] initWithFrame : frame];
      [self addSubview : spot];

      //ROOT's staff.
      controller = c;
      objectIndex = objIndex;

      const TObject *obj = controller.fileContainer->GetObject(objIndex);
      self.objectName = [NSString stringWithFormat : @"%s", obj->GetName()];
      self.icon = thumbnail;

      self.opaque = NO;

      //Tap gesture to select an object.
      UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(handleTap)];
      [self addGestureRecognizer : tap];
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect)rect
{
   CGContextRef ctx = UIGraphicsGetCurrentContext();

   if (isDirectory) {
      //Directory's icon is 128 x 128 < than thumbnail.
      CGPoint topLeft = CGPointMake([ObjectShortcut iconWidth] / 2 - folderIconSize.width / 2, [ObjectShortcut iconHeight] / 2 - folderIconSize.height / 2);
      [icon drawAtPoint : topLeft];
   } else
      [icon drawAtPoint : CGPointZero];

   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   const CGRect textRect = CGRectMake(0.f, [ObjectShortcut iconHeight], [ObjectShortcut iconWidth], [ObjectShortcut textHeight]);

#ifdef __IPHONE_6_0
   [objectName drawInRect : textRect withFont : [UIFont systemFontOfSize : 16] lineBreakMode : NSLineBreakByWordWrapping alignment : NSTextAlignmentCenter];
#else
   [objectName drawInRect : textRect withFont : [UIFont systemFontOfSize : 16] lineBreakMode : UILineBreakModeWordWrap alignment : UITextAlignmentCenter];
#endif
}

//____________________________________________________________________________________________________
- (void) handleTap
{
   [controller selectObjectFromFile : self];
}

@end
