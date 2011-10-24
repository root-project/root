#import <CoreGraphics/CGContext.h>
#import <QuartzCore/QuartzCore.h>

#import "RootFileController.h"
#import "FileShortcut.h"

//C++ (ROOT) imports.
#import "IOSFileContainer.h"

@implementation FileShortcut

@synthesize fileName;
@synthesize filePath;
@synthesize errorMessage;

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
- (void) initFileContainer : (NSString *) path
{
   //C++ part. Open the file and read its contents.
   fileContainer = ROOT::iOS::CreateFileContainer([path cStringUsingEncoding : [NSString defaultCStringEncoding]]);
}

//____________________________________________________________________________________________________
- (void) setPathAndName : (NSString *) path
{
   self.filePath = path;
   //extract file name substring.
   if (fileContainer)
      self.fileName = [NSString stringWithFormat : @"%s", fileContainer->GetFileName()];
}

//____________________________________________________________________________________________________
- (void) initImages
{
   filePictogram = [UIImage imageNamed : @"file_icon.png"];
   [filePictogram retain];
      
   backgroundImage = [UIImage imageNamed:@"file_shortcut_background.png"];
   [backgroundImage retain];
}

//____________________________________________________________________________________________________
- (void) initGestures 
{
   UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(handleTap)];
   [self addGestureRecognizer:tap];
   [tap release];
}

//____________________________________________________________________________________________________
- (id)initWithFrame:(CGRect)frame controller : (UIViewController *)c filePath : (NSString *)path
{
   self = [super initWithFrame : frame];
   
   if (self) {
      controller = c;

      [self initFileContainer : path];      
      [self setPathAndName : path];
      [self initImages];
      [self initGestures];
   
      self.opaque = NO;
   }
   
   return self;
}

//____________________________________________________________________________________________________
- (void)drawRect:(CGRect)rect
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

   self.fileName = nil;
   self.filePath = nil;
   
   [filePictogram release];
   [backgroundImage release];
   [super dealloc];
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
