#import <CoreGraphics/CGContext.h>
#import <QuartzCore/QuartzCore.h>
#import <Availability.h>

#import "RootFileController.h"
#import "FileShortcut.h"

//C++ imports.
#import "FileUtils.h"

@implementation FileShortcut {
   __weak UIViewController *controller;

   UIImage *filePictogram;

   ROOT::iOS::Browser::FileContainer *fileContainer;
}

@synthesize fileName;

//____________________________________________________________________________________________________
+ (CGFloat) iconWidth
{
   return 150.f;
}

//____________________________________________________________________________________________________
+ (CGFloat) textHeight
{
   return 50.f;
}

//____________________________________________________________________________________________________
+ (CGFloat) iconHeight
{
   return [FileShortcut iconWidth] + [FileShortcut textHeight];
}

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame controller : (UIViewController *)viewController fileContainer : (ROOT::iOS::Browser::FileContainer *)container;
{
   self = [super initWithFrame : frame];

   if (self) {
      controller = viewController;
      fileContainer = container;

      self.fileName = [NSString stringWithFormat : @"%s", fileContainer->GetFileName()];
      filePictogram = [UIImage imageNamed : @"file_icon.png"];
      UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget : self action : @selector(handleTap)];
      [self addGestureRecognizer : tap];
      UILongPressGestureRecognizer *longPress = [[UILongPressGestureRecognizer alloc] initWithTarget: self action:@selector(handleLongPress:)];
      [self addGestureRecognizer : longPress];

      self.opaque = NO;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect)rect
{
   // Drawing code
   CGContextRef ctx = UIGraphicsGetCurrentContext();

   //Draw the pictogram for ROOT's file.
   const CGPoint topLeftPicCorner = CGPointMake(rect.size.width / 2 - filePictogram.size.width / 2,
                                                (rect.size.height - [FileShortcut textHeight]) / 2 - filePictogram.size.height / 2);
   [filePictogram drawAtPoint:topLeftPicCorner];

   //Draw the file name.
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   const CGRect textRect = CGRectMake(0.f, [FileShortcut iconHeight] - [FileShortcut textHeight], [FileShortcut iconWidth], [FileShortcut textHeight]);

#ifdef __IPHONE_6_0
   [fileName drawInRect : textRect withFont : [UIFont systemFontOfSize : 16] lineBreakMode : NSLineBreakByWordWrapping alignment : NSTextAlignmentCenter];
#else
   [fileName drawInRect : textRect withFont : [UIFont systemFontOfSize : 16] lineBreakMode : UILineBreakModeWordWrap alignment : UITextAlignmentCenter];
#endif
}

//____________________________________________________________________________________________________
- (void)dealloc
{
   //Crazy name qualification :(
   ROOT::iOS::Browser::FileContainer::DeleteFileContainer(fileContainer);
}

//____________________________________________________________________________________________________
- (void) handleTap
{
   RootFileController *parentController = (RootFileController *)controller;
   [parentController fileWasSelected : self];
}

//____________________________________________________________________________________________________
- (void) handleLongPress : (UILongPressGestureRecognizer *)longPress
{
   if (longPress.state == UIGestureRecognizerStateBegan) {
      RootFileController *parentController = (RootFileController *)controller;
      [parentController tryToDelete : self];
   }
}

//____________________________________________________________________________________________________
- (ROOT::iOS::Browser::FileContainer *) getFileContainer
{
   return fileContainer;
}

@end
