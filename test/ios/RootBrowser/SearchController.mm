#import "SearchController.h"

@implementation SearchController {
   NSArray *visibleKeys;
}

@synthesize delegate;
@synthesize keys;

- (void) setKeys : (NSMutableArray *)k
{  
   keys = k;
   visibleKeys = k;
   
   [self.tableView reloadData];
}

- (void) viewDidLoad
{
   [super viewDidLoad];

   self.title = @"Objects and directories";
   self.contentSizeForViewInPopover = CGSizeMake(300.f, 280.f);
}


- (void) viewWillAppear : (BOOL)animated
{
 
   // Ensure the complete list of recents is shown on first display.
   [super viewWillAppear : animated];
}


- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation)interfaceOrientation
{
    return YES;
}


- (void)viewDidUnload
{
   [super viewDidUnload];
}


- (void) filterResultsUsingString : (NSString *) filterString
{
   // If the search string is zero-length, then restore the full list
   // otherwise create a predicate to filter the recent searches using the search string.
   
   if ([filterString length] == 0) {
      visibleKeys = keys;
   } else {
      NSPredicate *filterPredicate = [NSPredicate predicateWithFormat : @"self BEGINSWITH[cd] %@", filterString];
      visibleKeys = [keys filteredArrayUsingPredicate : filterPredicate];
   }

   [self.tableView reloadData];
}

#pragma mark Table view methods

- (NSInteger) tableView : (UITableView *)tableView numberOfRowsInSection : (NSInteger)section 
{    
   return [visibleKeys count];
}

- (UITableViewCell *) tableView : (UITableView *)tableView cellForRowAtIndexPath : (NSIndexPath *)indexPath
{
   UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier : @"Cell"];
   if (cell == nil)
      cell = [[UITableViewCell alloc] initWithStyle : UITableViewCellStyleDefault reuseIdentifier : @"Cell"];
   cell.textLabel.text = [visibleKeys objectAtIndex : indexPath.row];    
   return cell;
}


- (void) tableView : (UITableView *)tableView didSelectRowAtIndexPath : (NSIndexPath *)indexPath
{
   // Notify the delegate if a row is selected.
   [delegate searchesController : self didSelectString : [visibleKeys objectAtIndex:indexPath.row]];
}

@end
