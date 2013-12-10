#import "FileContainerElement.h"
#import "SearchViewController.h"

@implementation SearchViewController {
   NSArray *visibleKeys;
}

@synthesize delegate;
@synthesize keys;

//____________________________________________________________________________________________________
- (void) setKeys : (NSMutableArray *)k
{  
   keys = k;
   visibleKeys = k;
   
   [self.tableView reloadData];
}

//____________________________________________________________________________________________________
- (void) viewDidLoad
{
   [super viewDidLoad];

   self.title = @"Objects and directories";
   self.preferredContentSize = CGSizeMake(600.f, 280.f);
}

//____________________________________________________________________________________________________
- (void) viewWillAppear : (BOOL)animated
{
 
   // Ensure the complete list of recents is shown on first display.
   [super viewWillAppear : animated];
}

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
    return YES;
}

//____________________________________________________________________________________________________
- (void) filterResultsUsingString : (NSString *) filterString
{
   // If the search string is zero-length, then restore the full list
   // otherwise create a predicate to filter the recent searches using the search string.
   
   if ([filterString length] == 0) {
      visibleKeys = keys;
   } else {
      NSPredicate *filterPredicate = [NSPredicate predicateWithFormat : @"self.elementName BEGINSWITH[cd] %@", filterString];
      visibleKeys = [keys filteredArrayUsingPredicate : filterPredicate];
   }

   [self.tableView reloadData];
}

#pragma mark Table view methods

//____________________________________________________________________________________________________
- (NSInteger) tableView : (UITableView *)tableView numberOfRowsInSection : (NSInteger)section 
{    
   return [visibleKeys count];
}

//____________________________________________________________________________________________________
- (UITableViewCell *) tableView : (UITableView *)tableView cellForRowAtIndexPath : (NSIndexPath *)indexPath
{
   UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier : @"Cell"];
   if (cell == nil)
      cell = [[UITableViewCell alloc] initWithStyle : UITableViewCellStyleDefault reuseIdentifier : @"Cell"];
   
   FileContainerElement *key = (FileContainerElement *)[visibleKeys objectAtIndex : indexPath.row];
   cell.textLabel.text = key.elementName;
   return cell;
}

//____________________________________________________________________________________________________
- (void) tableView : (UITableView *)tableView didSelectRowAtIndexPath : (NSIndexPath *)indexPath
{
   // Notify the delegate if a row is selected.
   [delegate searchController : self didSelectKey : (FileContainerElement *)[visibleKeys objectAtIndex : indexPath.row]];
}

@end
