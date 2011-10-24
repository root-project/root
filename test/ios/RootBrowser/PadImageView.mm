#import "PadImageView.h"

@implementation PadImageView

@synthesize zoomed;

//____________________________________________________________________________________________________
- (void) dealloc
{
   [padImage release];
   [super dealloc];
}

//____________________________________________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   if (padImage)
      [padImage drawInRect : rect];
}

//____________________________________________________________________________________________________
- (void) setPadImage : (UIImage *)image
{
   if (image != padImage) {
      [padImage release];
      padImage = [image retain];
   }
}

//____________________________________________________________________________________________________
- (UIImage *) padImage
{
   return padImage;
}

@end
