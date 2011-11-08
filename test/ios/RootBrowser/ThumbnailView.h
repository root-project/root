#import <UIKit/UIKit.h>

@class ThumbnailView;

@protocol ThumbnailViewDelegate <NSObject>
@required
- (unsigned) numberOfThumbnailsInView : (ThumbnailView *)view;
- (UIView *) thumbnailAtIndex : (unsigned)index;
@optional
- (void) cacheDataForThumbnail : (UIView *)view;
- (void) loadDataForVisibleRange;
@end

/*

View with nested scroll-view with thumbnails.
Thumbnails are created by delegate.
Real data for thumbnail is loaded by delegate.
View places these thumbnails in a grid,
durign scrolling thumbnails, which became invisible,
are recycled - used for thumbnails, which became visible.
Thumbnails can contain some generic images, when scroll stopped,
delegate can load real data and create real images.
Delegate can cache images for previously shown thumbnails,
if view scrolls back, such cached images can be used instead
of loading.

ThumbnailView must be created with initWithFrame method.
Thumbnail's size (width) + addW must be < scroll.width - 2 * addSide.

This class was inspired by ATArrayView (Andrey Tarantsov, http://tarantsov.com/)
*/

@interface ThumbnailView : UIView

@property (nonatomic, weak) id<ThumbnailViewDelegate> delegate;
@property (nonatomic, assign) CGFloat addSide;
@property (nonatomic, assign) CGFloat addTop;
@property (nonatomic, assign) CGFloat addW;
@property (nonatomic, assign) CGFloat addH;
@property (nonatomic, assign) CGSize itemSize;

@property (nonatomic, readonly) unsigned firstVisibleThumbnail;
@property (nonatomic, readonly) unsigned lastVisibleThumbnail;

- (void) loadData;
- (UIView *) getThumbnailFromCache;

@end
