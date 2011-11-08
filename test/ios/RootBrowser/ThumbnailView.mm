#import "ThumbnailView.h"


const int preloadRows = 2;

@interface ThumbnailView () <UIScrollViewDelegate> {
   UIScrollView *scroll;

   NSMutableSet *visibleThumbnails;
   NSMutableSet *cachedThumbnails;
   
   //Grid parameters.
   unsigned nItems;
   unsigned nCols;
   unsigned nRows;
   
   CGFloat addX;//addSide + remaining width.
}

- (void) calculateGridParameters;
- (void) placeThumbnail : (UIView *)thumbnail;
- (void) placeThumbnails : (BOOL)fixPositions;

- (CGRect) frameForThumbnailAtIndex : (unsigned)index;
- (UIView *) thumbnailWithTag : (unsigned)tag;
- (void) cacheThumbnail : (UIView *)thumbnail;

@end

@implementation ThumbnailView

@synthesize delegate;
@synthesize addSide;
@synthesize addTop;
@synthesize addW;
@synthesize addH;
@synthesize itemSize;

#pragma mark - Initialization.

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame
{
   //View MUST be initialized using this method.
   if (self = [super initWithFrame : frame]) {
      //Grid. No data load yet.
      nItems = 0;
      nRows = 0;
      nCols = 0;
      //Some "default" values for grid's geometry,
      //can be modified using properties.
      addTop = 50.f;
      addSide = 50.f;
      addW = 50.f;
      addH = 50.f;
      itemSize = CGSizeMake(150.f, 150.f);

      visibleThumbnails = [[NSMutableSet alloc] init];
      cachedThumbnails = [[NSMutableSet alloc] init];
      
      //Setup nested scrollview.
      frame.origin = CGPointZero;

      scroll = [[UIScrollView alloc] initWithFrame : frame];
      scroll.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight | UIViewAutoresizingFlexibleLeftMargin | 
                                UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleBottomMargin;
      scroll.contentMode = UIViewContentModeScaleToFill;
      scroll.showsVerticalScrollIndicator = YES;
      scroll.showsHorizontalScrollIndicator = NO;
      scroll.delegate = self;
      [self addSubview : scroll];
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) loadData
{
   nItems = [delegate numberOfThumbnailsInView : self];
   
   for (UIView *v in visibleThumbnails)
      [self cacheThumbnail : v];

   [visibleThumbnails removeAllObjects];
   
   if (nItems) {
      [self calculateGridParameters];
      [self placeThumbnails : YES];
   }
}

#pragma mark - Geometry management.

//____________________________________________________________________________________________________
- (void) layoutSubviews
{
   if (nItems) {
      [self calculateGridParameters];
      [self placeThumbnails : YES];
   }
}

#pragma mark - Grid's management.

//____________________________________________________________________________________________________
- (void) showFirstLast
{
   NSLog(@"first %d last %d", self.firstVisibleThumbnail, self.lastVisibleThumbnail);
}

//____________________________________________________________________________________________________
- (void) setItemSize : (CGSize)newItemSize
{
   if (newItemSize.width + addW > scroll.bounds.size.width - 2 * addSide) {
      NSLog(@"ThumbnailView -setItemSize, item size is too big");
      exit(1);//Must be an exception. Check this later.
   }
   
   itemSize = newItemSize;
}

//____________________________________________________________________________________________________
- (unsigned) firstVisibleThumbnail
{
   const int firstRow = (scroll.bounds.origin.y - addTop) / (itemSize.height + addH) - preloadRows;
   if (firstRow < 0)
      return 0;

   return firstRow * nCols;
}

//____________________________________________________________________________________________________
- (unsigned) lastVisibleThumbnail
{
   //Pre-condition: nItems > 0, nCols > 0.
   const int lastRow = (CGRectGetMaxY(scroll.bounds) - addTop) / (itemSize.height + addH) + 1 + preloadRows;
   if (lastRow < 0)
      return 0;

   if (lastRow * nCols > nItems)
      return nItems - 1;

   return lastRow * nCols - 1;
}

//____________________________________________________________________________________________________
- (void) calculateGridParameters
{
   //Pre-condition: scroll width must be big enough to
   //position at least 1 thumbnail + additional spaces.
   const CGSize scrollSize = scroll.bounds.size;
   
   if (scrollSize.width - addSide * 2 < itemSize.width + addW) {
      //I do not know, if Apple's code can somehow set the bounds to 
      //be so small.
      NSLog(@"scroll is to small to place any thumbnail of required size");
      exit(1);
   }

   nCols = (scrollSize.width - addSide * 2) / (itemSize.width + addW);
   nRows = (nItems + nCols - 1) / nCols;
   addX = (scrollSize.width - nCols * (itemSize.width + addW)) / 2;
}

//____________________________________________________________________________________________________
- (void) placeThumbnail : (UIView *)thumbnail
{
   thumbnail.frame = [self frameForThumbnailAtIndex : thumbnail.tag];
   [thumbnail setNeedsDisplay];
}

//____________________________________________________________________________________________________
- (void) placeThumbnails : (BOOL) fixPos
{
   scroll.contentSize = CGSizeMake(scroll.frame.size.width, addTop * 2 + (itemSize.height + addH) * nRows - addH);

   const unsigned first = self.firstVisibleThumbnail;
   const unsigned last = self.lastVisibleThumbnail;	
   
   for (UIView *thumbnail in visibleThumbnails) {
      if (thumbnail.tag < first || thumbnail.tag > last) {
         //Thumbnail became invisible, remove it from scroll,
         //move it into cache, cache image data.
         [self cacheThumbnail : thumbnail];
      }
   }
   
   //Remove now invisible thumbnails.
   [visibleThumbnails minusSet : cachedThumbnails];
   
   //Position visible thumbnails.
   for (unsigned tag = first; tag <= last; ++tag) {
      UIView *thumbnail = [self thumbnailWithTag : tag];
      if (!thumbnail) {
         thumbnail = [delegate thumbnailAtIndex : tag];
         thumbnail.tag = tag;
         [scroll addSubview : thumbnail];
         [visibleThumbnails addObject : thumbnail];
      } else if (!fixPos)//For example, during scroll, no need to update anything.
         continue;
      
      [self placeThumbnail : thumbnail];
   }
}

//____________________________________________________________________________________________________
- (CGRect) frameForThumbnailAtIndex : (unsigned)index
{
   const unsigned row = index / nCols;
   const unsigned col = index % nCols;
   
   CGRect frame = CGRectZero;
   frame.origin.x = addX + (itemSize.width + addW) * col + 0.5 * addW;
   frame.origin.y = addTop + (itemSize.height + addH) * row;
   frame.size = itemSize;
   
   return frame;
}

#pragma mark - Thumbnails caching.

//____________________________________________________________________________________________________
- (UIView *) getThumbnailFromCache
{
   UIView *thumbnail = (UIView *)[cachedThumbnails anyObject];
   if (thumbnail)
      [cachedThumbnails removeObject : thumbnail];
      
   return thumbnail;
}

//____________________________________________________________________________________________________
- (UIView *) thumbnailWithTag : (unsigned)tag
{
   for (UIView * view in visibleThumbnails) {
      if (view.tag == tag)
         return view;
   }
   
   return nil;
}

//____________________________________________________________________________________________________
- (void) cacheThumbnail : (UIView *)thumbnail
{
   [cachedThumbnails addObject : thumbnail];
   
   if ([delegate respondsToSelector : @selector(cacheDataForThumbnail:)])
      [delegate cacheDataForThumbnail : thumbnail];//Save icon in a cache.

   [thumbnail removeFromSuperview];
}

#pragma mark - Scrollview delegate.

//____________________________________________________________________________________________________
- (void) loadDataForVisibleRange
{
   if ([delegate respondsToSelector : @selector(loadDataForVisibleRange)])
      [delegate loadDataForVisibleRange];
}

//____________________________________________________________________________________________________
- (void) scrollViewDidScroll : (UIScrollView *)scrollView
{
   [self placeThumbnails : NO];
   [self showFirstLast];
}

//____________________________________________________________________________________________________
- (void) scrollViewDidEndDecelerating:(UIScrollView *)scrollView
{
   //Now, ask delegate to really load objects.
   [self loadDataForVisibleRange];
}

//____________________________________________________________________________________________________
- (void) scrollViewDidEndDragging : (UIScrollView *)scrollView willDecelerate : (BOOL)decelerate
{
   if (!decelerate) {
      //Load objects.
      [self loadDataForVisibleRange];
   }
}

@end
