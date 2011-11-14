#import <UIKit/UIKit.h>

@class SearchController;

@protocol SearchDelegate
- (void) searchesController : (SearchController *)controller didSelectString : (NSString *)searchString;
@end


@interface SearchController : UITableViewController

@property (nonatomic, weak) id<SearchDelegate> delegate;
@property (nonatomic, retain) NSMutableArray *keys;

- (void) filterResultsUsingString : (NSString *)filterString;


@end
