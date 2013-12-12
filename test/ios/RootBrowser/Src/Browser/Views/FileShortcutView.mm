#import <cassert>

#import "FileCollectionViewController.h"
#import "FileShortcutView.h"

//C++ imports.
#import "FileUtils.h"

@implementation FileShortcutView {
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
   return [FileShortcutView iconWidth] + [FileShortcutView textHeight];
}

//____________________________________________________________________________________________________
- (instancetype) initWithFrame : (CGRect) frame controller : (UIViewController *) viewController
       fileContainer : (ROOT::iOS::Browser::FileContainer *) container;
{
   assert(viewController != nil && "initWithFrame:controller:fileContainer:, parameter 'viewController' is nil");
   assert(container != nullptr && "initWithFrame:controller:fileContainer:, parameter 'container' is nil");

   if (self = [super initWithFrame : frame]) {
      //
      controller = viewController;
      fileContainer = container;
      //
      fileName = [NSString stringWithFormat : @"%s", fileContainer->GetFileName()];
      //
      filePictogram = [UIImage imageNamed : @"file_icon.png"];
      UITapGestureRecognizer * const tap = [[UITapGestureRecognizer alloc] initWithTarget : self action : @selector(handleTap)];
      [self addGestureRecognizer : tap];
      UILongPressGestureRecognizer * const longPress = [[UILongPressGestureRecognizer alloc] initWithTarget : self action : @selector(handleLongPress:)];
      [self addGestureRecognizer : longPress];

      self.opaque = NO;
   }
   
   return self;
}

//____________________________________________________________________________________________________
- (void) dealloc
{
   //Crazy name qualification :(
   ROOT::iOS::Browser::FileContainer::DeleteFileContainer(fileContainer);
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect) rect
{
   //Draw the pictogram for ROOT's file.
   const CGPoint topLeftPicCorner = CGPointMake(rect.size.width / 2 - filePictogram.size.width / 2, 
                                                (rect.size.height - [FileShortcutView textHeight]) / 2 - filePictogram.size.height / 2);
   [filePictogram drawAtPoint : topLeftPicCorner];
   
   UIFont * const font = [UIFont systemFontOfSize : 16];
   NSMutableParagraphStyle * const paragraphStyle = [[NSParagraphStyle defaultParagraphStyle] mutableCopy];
   paragraphStyle.lineBreakMode = NSLineBreakByWordWrapping;
   paragraphStyle.alignment = NSTextAlignmentCenter;
   NSDictionary * const attributes = @{NSFontAttributeName : font, NSParagraphStyleAttributeName : paragraphStyle,
                                       NSForegroundColorAttributeName : [UIColor whiteColor]};
   const CGRect textRect = CGRectMake(0.f, [FileShortcutView iconHeight] - [FileShortcutView textHeight], [FileShortcutView iconWidth], [FileShortcutView textHeight]);
   [fileName drawInRect : textRect withAttributes : attributes];
}

//____________________________________________________________________________________________________
- (void) handleTap 
{
   assert(controller != nil && "handleTap, controller is nil");
   [(FileCollectionViewController *)controller fileWasSelected : self];
}

//____________________________________________________________________________________________________
- (void) handleLongPress : (UILongPressGestureRecognizer *) longPress
{
   assert(longPress != nil && "handleLongPress:, parameter 'longPress' is nil");
   assert(controller != nil && "handleLongPress:, controller is nil");

   if (longPress.state == UIGestureRecognizerStateBegan)
      [(FileCollectionViewController *)controller tryToDelete : self];
}

//____________________________________________________________________________________________________
- (ROOT::iOS::Browser::FileContainer *) getFileContainer
{
   return fileContainer;
}

@end
