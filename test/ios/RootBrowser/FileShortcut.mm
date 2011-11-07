#import <CoreGraphics/CGContext.h>
#import <QuartzCore/QuartzCore.h>

#import "RootFileController.h"
#import "FileShortcut.h"

//C++ (ROOT) imports.
#import "IOSFileContainer.h"

@implementation FileShortcut {
   __weak UIViewController *controller;

   UIImage *filePictogram;
   UIImage *backgroundImage;
   
   ROOT::iOS::FileContainer *fileContainer;
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
- (id) initWithFrame : (CGRect)frame controller : (UIViewController *)viewController fileContainer : (ROOT::iOS::FileContainer *)container;
{
   self = [super initWithFrame : frame];
   
   if (self) {
      controller = viewController;
      fileContainer = container;
      
      self.fileName = [NSString stringWithFormat : @"%s", fileContainer->GetFileName()];
      filePictogram = [UIImage imageNamed : @"file_icon.png"];
      backgroundImage = [UIImage imageNamed : @"file_shortcut_background.png"];
      UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(handleTap)];
      [self addGestureRecognizer:tap];
   
      self.opaque = NO;
   }
   
   return self;
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect)rect
{
   // Drawing code
   CGContextRef ctx = UIGraphicsGetCurrentContext();

   [backgroundImage drawAtPoint:CGPointZero];

   //Draw the pictogram for ROOT's file.
   const CGPoint topLeftPicCorner = CGPointMake(rect.size.width / 2 - filePictogram.size.width / 2, 
                                                (rect.size.height - [FileShortcut textHeight]) / 2 - filePictogram.size.height / 2);
   [filePictogram drawAtPoint:topLeftPicCorner];
   
   //Draw the file name.
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   const CGRect textRect = CGRectMake(0.f, [FileShortcut iconHeight] - [FileShortcut textHeight], [FileShortcut iconWidth], [FileShortcut textHeight]);
   [fileName drawInRect : textRect withFont : [UIFont systemFontOfSize : 16] lineBreakMode : UILineBreakModeWordWrap alignment : UITextAlignmentCenter];
}

//____________________________________________________________________________________________________
- (void)dealloc
{
   ROOT::iOS::DeleteFileContainer(fileContainer);
}

//____________________________________________________________________________________________________
- (void) handleTap 
{
   RootFileController *parentController = (RootFileController *)controller;
   [parentController fileWasSelected : self];
}

//____________________________________________________________________________________________________
- (ROOT::iOS::FileContainer *) getFileContainer
{
   return fileContainer;
}

@end
