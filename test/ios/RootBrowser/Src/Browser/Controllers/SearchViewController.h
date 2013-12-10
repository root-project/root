#import <UIKit/UIKit.h>

@class FileContainerElement;
@class SearchViewController;

@protocol SearchViewDelegate
- (void) searchController : (SearchViewController *) controller didSelectKey : (FileContainerElement *) key;
@end


@interface SearchViewController : UITableViewController

@property (nonatomic, weak) id<SearchViewDelegate> delegate;
@property (nonatomic) NSMutableArray *keys;

- (void) filterResultsUsingString : (NSString *) filterString;

@end


