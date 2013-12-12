#import <cassert>

#import "FileContentViewController.h"
#import "ObjectShortcutView.h"
#import "SpotObjectView.h"

//C++ (ROOT) imports.
#import "FileUtils.h"
#import "TObject.h"

const CGSize folderIconSize = CGSizeMake(128.f, 128.f);

@implementation ObjectShortcutView  {
   __weak FileContentViewController *controller;
   
   NSString *objectName;
   UIImage *icon;
}

@synthesize isDirectory;
@synthesize objectIndex;
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
   return CGRectMake(0.f, 0.f, [ObjectShortcutView iconWidth], [ObjectShortcutView iconHeight] + [ObjectShortcutView textHeight]);
}

//____________________________________________________________________________________________________
- (instancetype) initWithFrame : (CGRect) frame controller : (FileContentViewController*) c forFolderAtIndex : (unsigned) index
{
   assert(c != nil && "initWithFrame:controller:forFolderAtIndex:, parameter 'c' is nil");
   assert(c.fileContainer != nullptr && "initWithFrame:controller:forFolderAtIndex:, fileContainer is null");
   
   using namespace ROOT::iOS::Browser;
   
   if (self = [super initWithFrame : frame]) {
      frame.origin = CGPointZero;
      frame.size.height = [ObjectShortcutView iconHeight];
      
      spot = [[SpotObjectView alloc] initWithFrame : frame];
      [self addSubview : spot];

      controller = c;
      objectIndex = index;
      
      const FileContainer *cont = controller.fileContainer->GetDirectory(index);
      isDirectory = YES;
      objectName = [NSString stringWithFormat : @"%s", cont->GetFileName()];
      icon = [UIImage imageNamed : @"directory.png"];
      self.opaque = NO;
      
      //Tap gesture to select a directory.
      UITapGestureRecognizer * const tap = [[UITapGestureRecognizer alloc] initWithTarget : self action : @selector(handleTap)];
      [self addGestureRecognizer : tap];
   }
   
   return self;
}

//____________________________________________________________________________________________________
- (instancetype) initWithFrame : (CGRect) frame controller : (FileContentViewController*) c forObjectAtIndex : (unsigned) objIndex withThumbnail : (UIImage *) thumbnail
{
   assert(c != nil && "initWithFrame:controller:forObjectAtIndex:withThumbnail:, parameter 'c' is nil");
   assert(c.fileContainer != nullptr && "initWithFrame:controller:forObjectAtIndex:withThumbnail:, fileContainer is null");

   using namespace ROOT::iOS::Browser;

   if (self = [super initWithFrame : frame]) {
      frame.origin = CGPointZero;
      frame.size.height = [ObjectShortcutView iconHeight];
      
      spot = [[SpotObjectView alloc] initWithFrame : frame];
      [self addSubview : spot];
   
      //ROOT's staff.
      controller = c;
      objectIndex = objIndex;
      
      const TObject *obj = controller.fileContainer->GetObject(objIndex);
      isDirectory = NO;
      objectName = [NSString stringWithFormat : @"%s", obj->GetName()];
      icon = thumbnail;
   
      self.opaque = NO;
      
      //Tap gesture to select an object.
      UITapGestureRecognizer * const tap = [[UITapGestureRecognizer alloc] initWithTarget : self action : @selector(handleTap)];
      [self addGestureRecognizer : tap];
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect) rect
{
   if (isDirectory) {
      //Directory's icon is 128 x 128 < than thumbnail.
      CGPoint topLeft = CGPointMake([ObjectShortcutView iconWidth] / 2 - folderIconSize.width / 2, [ObjectShortcutView iconHeight] / 2 - folderIconSize.height / 2);
      [icon drawAtPoint : topLeft];   
   } else
      [icon drawAtPoint : CGPoint()];

   //
   UIFont * const font = [UIFont systemFontOfSize : 16];
   NSMutableParagraphStyle * const paragraphStyle = [[NSParagraphStyle defaultParagraphStyle] mutableCopy];
   paragraphStyle.lineBreakMode = NSLineBreakByWordWrapping;
   paragraphStyle.alignment = NSTextAlignmentCenter;
   NSDictionary * const attributes = @{NSFontAttributeName : font, NSParagraphStyleAttributeName : paragraphStyle,
                                       NSForegroundColorAttributeName : [UIColor whiteColor]};
   //
   const CGRect textRect = CGRectMake(0.f, [ObjectShortcutView iconHeight], [ObjectShortcutView iconWidth], [ObjectShortcutView textHeight]);
   [objectName drawInRect : textRect withAttributes : attributes];
}

//____________________________________________________________________________________________________
- (void) handleTap
{
   [controller selectObjectFromFile : self];
}

@end
