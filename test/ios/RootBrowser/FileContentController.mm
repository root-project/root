#import <stdlib.h>

#import <CoreGraphics/CGGeometry.h>
#import <QuartzCore/QuartzCore.h>

#import "FileContentController.h"
#import "ROOTObjectController.h"
#import "FileContainerElement.h"
#import "SlideshowController.h"
#import "TransparentToolbar.h"
#import "SearchController.h"
#import "ObjectShortcut.h"
#import "Shortcuts.h"


//C++ imports.
#import "IOSPad.h"

#import "FileUtils.h"

@interface FileContentController () {
   NSMutableArray *objectShortcuts;
   UISearchBar *searchBar;
   UIPopoverController *searchPopover;
   SearchController *searchController;
   UIBarButtonItem *slideShowBtn;
   
   BOOL animateDirAfterLoad;
   BOOL animateObjAfterLoad;
   unsigned spotElement;
}

- (void) highlightDirectory : (unsigned)tag;
- (void) highlightObject : (unsigned)tag;

@end


@implementation FileContentController

@synthesize fileContainer;


//____________________________________________________________________________________________________
- (void) initToolbarItems
{
   UIToolbar *toolbar = [[TransparentToolbar alloc] initWithFrame : CGRectMake(0.f, 0.f, 250.f, 44.f)];
   toolbar.barStyle = UIBarStyleBlackTranslucent;

   NSMutableArray *items = [[NSMutableArray alloc] initWithCapacity : 2];
   
   searchBar = [[UISearchBar alloc] initWithFrame:CGRectMake(0.f, 0.f, 150.f, 44.f)];
   searchBar.delegate = self;

   UIBarButtonItem *searchItem = [[UIBarButtonItem alloc] initWithCustomView : searchBar];
   [items addObject : searchItem];

   slideShowBtn = [[UIBarButtonItem alloc] initWithTitle : @"Slide show" style : UIBarButtonItemStyleBordered target : self action : @selector(startSlideshow)];
   [items addObject : slideShowBtn];
   
   [toolbar setItems : items animated : NO];
   
   UIBarButtonItem *rightItem = [[UIBarButtonItem alloc] initWithCustomView : toolbar];
   rightItem.style = UIBarButtonItemStylePlain;
   self.navigationItem.rightBarButtonItem = rightItem;
   
   
}

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];

   if (self) {
      [self view];
      [self initToolbarItems];
      searchController = [[SearchController alloc] initWithStyle : UITableViewStylePlain];
      searchController.delegate = self;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void)didReceiveMemoryWarning
{
   // Releases the view if it doesn't have a superview.
   [super didReceiveMemoryWarning];
    
   // Release any cached data, images, etc that aren't in use.
}

#pragma mark - View lifecycle


//____________________________________________________________________________________________________
- (void) correctFramesForOrientation : (UIInterfaceOrientation) orientation
{
   //TODO: all this staff with manual geometry management should be deleted, as soon as I switch to
   //ThumbnailView class.
   CGRect mainFrame;
   CGRect scrollFrame;

   if (orientation == UIInterfaceOrientationPortrait || orientation == UIInterfaceOrientationPortraitUpsideDown) {
      mainFrame = CGRectMake(0.f, 0.f, 768.f, 1004.f);
      scrollFrame = CGRectMake(0.f, 44.f, 768.f, 960.f);
   } else {
      mainFrame = CGRectMake(0.f, 0.f, 1024.f, 748.f);
      scrollFrame = CGRectMake(0.f, 44.f, 1024.f, 704.f);   
   }
   
   self.view.frame = mainFrame;
   scrollView.frame = scrollFrame;
   
   if ([[scrollView subviews] count])
      [ShorcutUtil placeShortcuts : objectShortcuts inScrollView : scrollView withSize : CGSizeMake([ObjectShortcut iconWidth], [ObjectShortcut iconHeight] + [ObjectShortcut textHeight]) andSpace : 100.f];
}

//____________________________________________________________________________________________________
- (void) viewWillAppear:(BOOL)animated
{
   //TODO: all this staff with manual geometry management should be deleted, as soon as I switch to
   //ThumbnailView class.
   //self.interfaceOrientation ?
   [self correctFramesForOrientation : [UIApplication sharedApplication].statusBarOrientation];
}

//____________________________________________________________________________________________________
- (void) viewDidLoad
{
   [super viewDidLoad];
   // Do any additional setup after loading the view from its nib.
}

//____________________________________________________________________________________________________
- (void) viewDidAppear:(BOOL)animated
{
   [super viewDidAppear : animated];
   if (animateDirAfterLoad) {
      [self highlightDirectory : spotElement];
      animateDirAfterLoad = NO;
   } else if (animateObjAfterLoad) {
      [self highlightObject : spotElement];
      animateObjAfterLoad = NO;
   }
}

//____________________________________________________________________________________________________
- (void) viewDidUnload
{
   [super viewDidUnload];
   // Release any retained subviews of the main view.
   // e.g. self.myOutlet = nil;
}

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
   // Return YES for supported orientations
	return YES;
}

//____________________________________________________________________________________________________
- (void) willAnimateRotationToInterfaceOrientation : (UIInterfaceOrientation)interfaceOrientation duration:(NSTimeInterval)duration 
{
   //TODO: all this staff with manual geometry management should be deleted, as soon as I switch to
   //ThumbnailView class.

   [self correctFramesForOrientation : interfaceOrientation];
}

//____________________________________________________________________________________________________
- (void) didRotateFromInterfaceOrientation : (UIInterfaceOrientation)fromInterfaceOrientation 
{
   //Bring back the popover after rotating.
   if (searchPopover) {
      [searchPopover presentPopoverFromRect : searchBar.bounds inView : searchBar
      permittedArrowDirections : UIPopoverArrowDirectionAny animated : NO];
   }
}

//____________________________________________________________________________________________________
- (void) clearScrollview
{
   NSArray *viewsToRemove = [scrollView subviews];
   for (UIView *v in viewsToRemove)
      [v removeFromSuperview];

}

//____________________________________________________________________________________________________
- (void) addShortcutForObjectAtIndex : (unsigned) objIndex
{
   const CGRect rect = CGRectMake(0.f, 0.f, [ObjectShortcut iconWidth], [ObjectShortcut iconHeight]);
   UIGraphicsBeginImageContext(rect.size);
   CGContextRef ctx = UIGraphicsGetCurrentContext();
   if (!ctx) {
      UIGraphicsEndImageContext();
      return;
   }
      
   //Now draw into this context.
   CGContextTranslateCTM(ctx, 0.f, rect.size.height);
   CGContextScaleCTM(ctx, 1.f, -1.f);
      
   //Fill bitmap with white first.
   CGContextSetRGBFillColor(ctx, 0.f, 0.f, 0.f, 1.f);
   CGContextFillRect(ctx, rect);
   //Set context and paint pad's contents
   //with special colors (color == object's identity)
   ROOT::iOS::Pad *pad = fileContainer->GetPadAttached(objIndex);
   pad->cd();
   pad->SetViewWH(rect.size.width, rect.size.height);
   pad->SetContext(ctx);
   pad->PaintThumbnail();
   
   UIImage *thumbnailImage = UIGraphicsGetImageFromCurrentImageContext();//autoreleased UIImage.
   UIGraphicsEndImageContext();
       
   ObjectShortcut *shortcut = [[ObjectShortcut alloc] initWithFrame : [ObjectShortcut defaultRect] controller : self forObjectAtIndex:objIndex withThumbnail : thumbnailImage];
   shortcut.layer.shadowColor = [UIColor blackColor].CGColor;
   shortcut.layer.shadowOffset = CGSizeMake(20.f, 20.f);
   shortcut.layer.shadowOpacity = 0.3f;

   [scrollView addSubview : shortcut];
   [objectShortcuts addObject : shortcut];

   UIBezierPath *path = [UIBezierPath bezierPathWithRect : rect];
   shortcut.layer.shadowPath = path.CGPath;
}

//____________________________________________________________________________________________________
- (void) addShortcutForFolderAtIndex : (unsigned) index
{
   ObjectShortcut *shortcut = [[ObjectShortcut alloc] initWithFrame : [ObjectShortcut defaultRect] controller : self forFolderAtIndex : index];
   [scrollView addSubview : shortcut];
   [objectShortcuts addObject : shortcut];
}

//____________________________________________________________________________________________________
- (void) addObjectsIntoScrollview
{
   using namespace ROOT::iOS::Browser;

   [self clearScrollview];

   objectShortcuts = [[NSMutableArray alloc] init];

   //Add directories first.
   for (FileContainer::size_type i = 0; i < fileContainer->GetNumberOfDirectories(); ++i)
      [self addShortcutForFolderAtIndex : i];
   //Now add objects.
   for (FileContainer::size_type i = 0; i < fileContainer->GetNumberOfObjects(); ++i)
      [self addShortcutForObjectAtIndex : i];
}

//____________________________________________________________________________________________________
- (void) activateForFile : (ROOT::iOS::Browser::FileContainer *)container
{
   fileContainer = container;
   self.navigationItem.title = [NSString stringWithFormat : @"Contents of %s", container->GetFileName()];
   slideShowBtn.enabled = fileContainer->GetNumberOfObjects() > 1 ? YES : NO;
   
   //Prepare objects' thymbnails.
   [self addObjectsIntoScrollview];
   [self correctFramesForOrientation : [UIApplication sharedApplication].statusBarOrientation];
}

//____________________________________________________________________________________________________
- (void) startSlideshow
{
   SlideshowController *slideshowController = [[SlideshowController alloc] initWithNibName : @"SlideshowController" bundle : nil fileContainer : fileContainer];
   [self.navigationController pushViewController : slideshowController animated : YES];
}

//____________________________________________________________________________________________________
- (void) doTest
{
   const unsigned testIndex = 1 + rand() % (fileContainer->GetNumberOfObjects() - 1);
   ROOTObjectController *objectController = [[ROOTObjectController alloc] initWithNibName:@"ROOTObjectController" bundle : nil];
   [objectController setNavigationForObjectWithIndex : testIndex fromContainer : fileContainer];
   [self.navigationController pushViewController : objectController animated : YES];
}

//____________________________________________________________________________________________________
- (void) selectObjectFromFile : (ObjectShortcut *) shortcut
{
   if (shortcut.isDirectory) {
      //Create another FileContentController and push it on stack.
      FileContentController *contentController = [[FileContentController alloc] initWithNibName : @"FileContentController" bundle : nil];
      [contentController activateForFile : fileContainer->GetDirectory(shortcut.objectIndex)];
      [self.navigationController pushViewController : contentController animated : YES];
   } else {
      ROOTObjectController *objectController = [[ROOTObjectController alloc] initWithNibName : @"ROOTObjectController" bundle : nil];
      [objectController setNavigationForObjectWithIndex : shortcut.objectIndex fromContainer : fileContainer];
      [self.navigationController pushViewController : objectController animated : YES];
   }
}

#pragma mark - Search delegate.

//____________________________________________________________________________________________________
- (void) searchBarTextDidBeginEditing : (UISearchBar *)aSearchBar
{
   typedef ROOT::iOS::Browser::FileContainer::size_type size_type;

   if (auto nEntities = fileContainer->GetNumberOfDescriptors()) {
      if (!searchPopover) {         
         UINavigationController *navController = [[UINavigationController alloc] initWithRootViewController : searchController];
         searchPopover = [[UIPopoverController alloc] initWithContentViewController : navController];
         searchPopover.delegate = self;
         searchPopover.passthroughViews = [NSArray arrayWithObject : searchBar];
      }
      
      NSMutableArray *keys = [[NSMutableArray alloc] init];
      for (size_type i = 0; i < nEntities; ++i) {
         const auto &descriptor = fileContainer->GetElementDescriptor(i);
         NSString *formatString = descriptor.fIsDir ? @"%s (directory)" : @"%s";
         FileContainerElement *newKey = [[FileContainerElement alloc] init];
         newKey.elementName = [NSString stringWithFormat : formatString, descriptor.fName.c_str()];
         newKey.elementIndex = i;
         [keys addObject : newKey];
      }
      
      searchController.keys = keys;
      [searchPopover presentPopoverFromRect : [searchBar bounds] inView : searchBar permittedArrowDirections : UIPopoverArrowDirectionAny animated : YES];
   }
}

//____________________________________________________________________________________________________
- (void) searchBarTextDidEndEditing : (UISearchBar *)aSearchBar
{
   if (searchPopover) {
      [searchPopover dismissPopoverAnimated:YES];
      searchPopover = nil;
   }  

   [aSearchBar resignFirstResponder];
}

//____________________________________________________________________________________________________
- (void) searchBar : (UISearchBar *)searchBar textDidChange : (NSString *)searchText 
{
   // When the search string changes, filter the recents list accordingly.
   [searchController filterResultsUsingString : searchText];
}

//____________________________________________________________________________________________________
- (void) searchBarSearchButtonClicked : (UISearchBar *)aSearchBar 
{
   //NSLog(@"search clicked");
   [searchPopover dismissPopoverAnimated : YES];
   [searchBar resignFirstResponder];
}

#pragma mark - Popover controller delegate.

//____________________________________________________________________________________________________
- (void) popoverControllerDidDismissPopover : (UIPopoverController *)popoverController 
{
   //NSLog(@"popover dismiss");
   [searchBar resignFirstResponder];
}

#pragma mark - Search delegate.

//____________________________________________________________________________________________________
- (void) searchesController : (SearchController *)controller didSelectKey : (FileContainerElement *)key
{
   //NSLog(@"selected %@ with index %d", key.elementName, key.elementIndex);
   assert(key.elementIndex < fileContainer->GetNumberOfDescriptors());

   [searchPopover dismissPopoverAnimated : YES];
   searchPopover = nil;
   [searchBar resignFirstResponder];
   
   const auto &descriptor = fileContainer->GetElementDescriptor(key.elementIndex);
   if (descriptor.fOwner == fileContainer) {
      descriptor.fIsDir ? [self highlightDirectory : descriptor.fIndex] : [self highlightObject : descriptor.fIndex];
   } else {
      //Create another FileContentController and push it on stack.
      FileContentController *contentController = [[FileContentController alloc] initWithNibName : @"FileContentController" bundle : nil];
      [contentController activateForFile : descriptor.fOwner];

      if (descriptor.fIsDir)
         contentController->animateDirAfterLoad = YES;
      else
         contentController->animateObjAfterLoad = YES;
      
      contentController->spotElement = descriptor.fIndex;

      [self.navigationController pushViewController : contentController animated : YES];
   }
}

#pragma mark - adjust file container to show search result

//____________________________________________________________________________________________________
- (void) animateShortcut : (ObjectShortcut *) sh
{
   //Now, animation!
   CGAffineTransform originalTransform = sh.transform;
   CGAffineTransform newTransform = CGAffineTransformScale(originalTransform, 1.4f, 1.4f);
   [UIView beginAnimations : @"hide_object" context : nil];
   [UIView setAnimationDuration : 1.5f];
   [UIView setAnimationCurve : UIViewAnimationCurveLinear];
   [UIView setAnimationTransition : UIViewAnimationTransitionNone forView : sh cache : YES];
   sh.transform = newTransform;
   [UIView commitAnimations];

   [UIView beginAnimations : @"show_object" context : nil];
   [UIView setAnimationDuration : 1.f];
   [UIView setAnimationCurve : UIViewAnimationCurveLinear];
   [UIView setAnimationTransition : UIViewAnimationTransitionNone forView : sh cache : YES];
   sh.transform = originalTransform;
   [UIView commitAnimations];
}

//____________________________________________________________________________________________________
- (void) highlightDirectory : (unsigned)tag
{
   for (ObjectShortcut *sh in objectShortcuts) {
      if (sh.objectIndex == tag && sh.isDirectory) {
         const CGRect thumbFrame = sh.frame;
         const CGRect scrollBounds = scrollView.bounds;
         if (CGRectGetMaxY(thumbFrame) > CGRectGetMaxY(scrollBounds)) {
            //We have to scroll view to show object's or directory's shortcut.
            //Find new Y for bounds.
            const CGFloat newY = CGRectGetMaxY(thumbFrame) - scrollBounds.size.height;
            CGRect newBounds = scrollBounds;
            newBounds.origin.y = newY;
            [scrollView scrollRectToVisible : newBounds animated : YES];
         }
         
         [self animateShortcut : sh];

         break;
      }
   }
}

//____________________________________________________________________________________________________
- (void) highlightObject : (unsigned)tag
{
   for (ObjectShortcut *sh in objectShortcuts) {
      if (sh.objectIndex == tag && !sh.isDirectory) {
         CGRect thumbFrame = sh.frame;
         const CGRect scrollBounds = scrollView.bounds;
         if (CGRectGetMaxY(thumbFrame) > CGRectGetMaxY(scrollBounds)) {
            //We have to scroll view to show object's or directory's shortcut.
            //Find new Y for bounds.
            const CGFloat newY = CGRectGetMaxY(thumbFrame) - scrollBounds.size.height;
            CGRect newBounds = scrollBounds;
            newBounds.origin.y = newY;
            [scrollView scrollRectToVisible : newBounds animated : YES];
         }
         
         [self animateShortcut : sh];

         break;
      }
   }
}

@end
