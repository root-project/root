#import <UIKit/UIKit.h>

@class FileContainerElement;
@class SearchController;

@protocol SearchDelegate
- (void) searchesController : (SearchController *)controller didSelectKey : (FileContainerElement *)key;
@end


@interface SearchController : UITableViewController

@property (nonatomic, weak) id<SearchDelegate> delegate;
@property (nonatomic, retain) NSMutableArray *keys;

- (void) filterResultsUsingString : (NSString *)filterString;


@end
