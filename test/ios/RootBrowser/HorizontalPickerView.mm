#import <QuartzCore/QuartzCore.h>

#import "HorizontalPickerView.h"

namespace {
//Obj-C class is not a scope :((( UGLY LANGUAGE!
const CGFloat pickerWidth = 200.f;
const CGFloat cellWidth = 50.f;
const CGFloat cellHeight = 50.f;
const CGFloat xPad = 1.5 * cellWidth;
const CGFloat markerPos = 100.f;

}

@implementation HorizontalPickerView {
   UIScrollView *contentScroll;
   UIImageView *arrowView;
   UIImage *frameImage;
   UIImage *backgroundImage;
   
   unsigned selectedItem;
}


@synthesize pickerDelegate;

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame
{
   self = [super initWithFrame : frame];
   if (self) {
      self.backgroundColor = [UIColor clearColor];
   
      contentScroll = [[UIScrollView alloc] initWithFrame : CGRectMake(10.f, 10.f, pickerWidth, cellHeight)];
      contentScroll.scrollEnabled = YES;
      contentScroll.pagingEnabled = NO;
      contentScroll.delegate = self;
      contentScroll.showsVerticalScrollIndicator = NO;
      contentScroll.showsHorizontalScrollIndicator = NO;
      
      contentScroll.backgroundColor = [UIColor clearColor];
      [self addSubview : contentScroll];
      
      backgroundImage = [UIImage imageNamed : @"picker_bkn.png"];
      frameImage = [UIImage imageNamed : @"picker_frame_bkn.png"];
      
      CAGradientLayer *dropshadowLayer = [CAGradientLayer layer];
      dropshadowLayer.startPoint = CGPointMake(0.0f, 0.0f);
      dropshadowLayer.endPoint = CGPointMake(0.0f, 1.0f);
      dropshadowLayer.opacity = 1.0;
      dropshadowLayer.frame = CGRectMake(contentScroll.frame.origin.x, contentScroll.frame.origin.y, 
                                         contentScroll.frame.size.width, contentScroll.frame.size.height);
      dropshadowLayer.locations = [NSArray arrayWithObjects : [NSNumber numberWithFloat : 0.0f],
                                                              [NSNumber numberWithFloat : 0.05f],
                                                              [NSNumber numberWithFloat : 0.2f],
                                                              [NSNumber numberWithFloat : 0.8f],
                                                              [NSNumber numberWithFloat : 0.95f],                                   
                                                              [NSNumber numberWithFloat : 1.0f], nil];
      dropshadowLayer.colors = [NSArray arrayWithObjects : 
                                             (id)[[UIColor colorWithRed : 0.05f green : 0.05f blue : 0.05f alpha : 0.75f] CGColor], 
                                             (id)[[UIColor colorWithRed : 0.25f green : 0.25f blue : 0.25f alpha : 0.55f] CGColor], 
                                             (id)[[UIColor colorWithRed : 1.f green : 1.f blue : 1.f alpha : 0.05f] CGColor], 
                                             (id)[[UIColor colorWithRed : 1.f green : 1.f blue : 1.f alpha : 0.05f] CGColor], 
                                             (id)[[UIColor colorWithRed : 0.25f green : 0.25f blue : 0.25f alpha : 0.55f] CGColor],
                                             (id)[[UIColor colorWithRed : 0.05f green : 0.05f blue : 0.05f alpha : 0.75f] CGColor], nil];

      [self.layer insertSublayer:dropshadowLayer above : contentScroll.layer];
        
      CAGradientLayer *gradientLayer = [CAGradientLayer layer];
      gradientLayer.startPoint = CGPointMake(0.0f, 0.0f);
      gradientLayer.endPoint = CGPointMake(1.0f, 0.0f);
      gradientLayer.opacity = 1.0;
      gradientLayer.frame = CGRectMake(contentScroll.frame.origin.x, contentScroll.frame.origin.y, 
                                      contentScroll.frame.size.width, contentScroll.frame.size.height);
      gradientLayer.locations = [NSArray arrayWithObjects:
                                [NSNumber numberWithFloat:0.0f],
                                [NSNumber numberWithFloat:0.05f],
                                [NSNumber numberWithFloat:0.3f],
                                [NSNumber numberWithFloat:0.7f],
                                [NSNumber numberWithFloat:0.95f],                                   
                                [NSNumber numberWithFloat:1.0f], nil];
      gradientLayer.colors = [NSArray arrayWithObjects:
                             (id)[[UIColor colorWithRed:0.05f green:0.05f blue:0.05f alpha:0.95] CGColor], 
                             (id)[[UIColor colorWithRed:0.25f green:0.25f blue:0.25f alpha:0.8] CGColor], 
                             (id)[[UIColor colorWithRed:1.0f green:1.0f blue:1.0f alpha:0.1] CGColor], 
                             (id)[[UIColor colorWithRed:1.0f green:1.0f blue:1.0f alpha:0.1] CGColor], 
                             (id)[[UIColor colorWithRed:0.25f green:0.25f blue:0.25f alpha:0.8] CGColor],
                             (id)[[UIColor colorWithRed:0.05f green:0.05f blue:0.05f alpha:0.95] CGColor], nil];
      [self.layer insertSublayer:gradientLayer above:dropshadowLayer];

      arrowView = [[UIImageView alloc] initWithImage:[UIImage imageNamed:@"picker_arrow.png"]];
      arrowView.center = CGPointMake(frame.size.width / 2, 60 - arrowView.frame.size.height / 2);
      [self addSubview : arrowView];
      [self bringSubviewToFront : arrowView];
   }

   return self;
}

//____________________________________________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   [frameImage drawInRect : rect];
   [backgroundImage drawInRect:CGRectMake(10.f, 10.f, 200.f, 50.f)];
}

//____________________________________________________________________________________________________
- (void) adjustScroll
{
   CGPoint offset = contentScroll.contentOffset;
   const CGFloat currentPos = markerPos + offset.x - xPad;
   selectedItem = unsigned(currentPos / cellWidth);
   const CGFloat newPos = selectedItem * cellWidth + 0.5 * cellWidth;
   const CGFloat add = newPos - currentPos;
   offset.x += add;
   [contentScroll setContentOffset : offset animated : YES];
}

//____________________________________________________________________________________________________
- (void) setSelectedItem:(unsigned int)item
{
   selectedItem = item;
   const CGFloat x = xPad + selectedItem * cellWidth + 0.5f * cellWidth - markerPos;
   contentScroll.contentOffset = CGPointMake(x, 0.f);
}

//____________________________________________________________________________________________________
- (void) notify
{
   [pickerDelegate item : selectedItem wasSelectedInPicker : self];
}

//____________________________________________________________________________________________________
- (void) scrollViewDidEndDecelerating : (UIScrollView *) sender
{
   [self adjustScroll];
   [self notify];
}

- (void)scrollViewDidEndDragging:(UIScrollView *)scrollView willDecelerate:(BOOL)decelerate
{
   if (!decelerate) {
      [self adjustScroll];
      [self notify];
   }
}

#pragma mark - Picker's content management.

//____________________________________________________________________________________________________
- (void) addItems : (NSMutableArray *)items
{
   NSEnumerator *enumerator = [items objectEnumerator];
   UIView *v = [enumerator nextObject];
   for (unsigned i = 0; v; v = [enumerator nextObject], ++i) {
      //Adjust view position inside a scroll:
      const CGRect viewFrame = CGRectMake(i * cellWidth + xPad, 0.f, cellWidth, cellHeight);
      v.frame = viewFrame;
      [contentScroll addSubview : v];
   }

   contentScroll.contentSize = CGSizeMake(2 * xPad + [items count] * cellWidth, cellHeight);
}

@end
