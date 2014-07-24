#import <cassert>
#import <cmath>

#import <QuartzCore/QuartzCore.h>

#import "FileContentViewController.h"
#import "SlideshowViewController.h"
#import "ObjectViewController.h"
#import "FileContainerElement.h"
#import "TransparentToolbar.h"
#import "ObjectShortcutView.h"
#import "SpotObjectView.h"
#import "Shortcuts.h"
#import "Constants.h"

//C++ imports.
#import "IOSPad.h"

#import "FileUtils.h"

@implementation FileContentViewController {
   ROOT::iOS::Browser::FileContainer *fileContainer;
   
   __weak IBOutlet UIScrollView *scrollView;

   NSMutableArray *objectShortcuts;
   
   UISearchBar *searchBar;
   UIPopoverController *searchPopover;
   SearchViewController *searchController;
   
   UIBarButtonItem *slideShowBtn;
   
   BOOL animateDirAfterLoad;
   BOOL animateObjAfterLoad;

   unsigned spotElement;
   
   BOOL viewDidAppear;
   
   BOOL animating;
}

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
- (instancetype) initWithCoder : (NSCoder *) aDecoder
{
   if (self = [super initWithCoder : aDecoder]) {
      //
      viewDidAppear = NO;
      animating = NO;
   }
   
   return self;
}

#pragma mark - View lifecycle

//____________________________________________________________________________________________________
- (void) viewWillAppear : (BOOL) animated
{
   [super viewWillAppear : animated];
   [self correctFramesForOrientation : self.interfaceOrientation];
}

//____________________________________________________________________________________________________
- (void) viewDidLoad
{
   [super viewDidLoad];
   //
   [self initToolbarItems];
   searchController = [[SearchViewController alloc] initWithStyle : UITableViewStylePlain];
   searchController.delegate = self;
   //
   assert(fileContainer != nil && "viewDidLoad, fileContainer is nil");
   //Create object shortcuts.
   self.navigationItem.title = [NSString stringWithFormat : @"Contents of %s", fileContainer->GetFileName()];
   slideShowBtn.enabled = fileContainer->GetNumberOfObjects() > 1 ? YES : NO;
   [self addObjectsIntoScrollview];
}

//____________________________________________________________________________________________________
- (void) viewDidAppear : (BOOL)animated
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
- (void) viewDidLayoutSubviews
{
   [self correctFramesForOrientation : self.interfaceOrientation];
}

#pragma mark - Views' geometry + interface orientation.

//____________________________________________________________________________________________________
- (void) correctFramesForOrientation : (UIInterfaceOrientation) orientation
{
#pragma unused(orientation)

   //It's a legacy code - in the past I was resetting view's geometry manually.
   //Now it's done with automatic layout + I'm setting shortcuts myself.

   using ROOT::iOS::Browser::PlaceShortcutsInScrollView;
   
   if ([[scrollView subviews] count]) {
      PlaceShortcutsInScrollView(objectShortcuts, scrollView,
                                 CGSizeMake([ObjectShortcutView iconWidth], [ObjectShortcutView iconHeight] + [ObjectShortcutView textHeight]),
                                 100.f);
   }
}


//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)
   return YES;
}

//____________________________________________________________________________________________________
- (void) willAnimateRotationToInterfaceOrientation : (UIInterfaceOrientation)interfaceOrientation duration : (NSTimeInterval) duration
{
#pragma unused(duration)
   [self correctFramesForOrientation : interfaceOrientation];
}

//____________________________________________________________________________________________________
- (void) didRotateFromInterfaceOrientation : (UIInterfaceOrientation) fromInterfaceOrientation
{
#pragma unused(fromInterfaceOrientation)
   //Bring back the popover after rotating.
   if (searchPopover) {
      [searchPopover presentPopoverFromRect : searchBar.bounds inView : searchBar
      permittedArrowDirections : UIPopoverArrowDirectionAny animated : NO];
   }
}

#pragma mark - objects and folders (shortcut views).

//____________________________________________________________________________________________________
- (void) clearScrollview
{
   NSArray * const viewsToRemove = [scrollView subviews];
   for (UIView *v in viewsToRemove)
      [v removeFromSuperview];
}

//____________________________________________________________________________________________________
- (void) addShortcutForObjectAtIndex : (unsigned) objIndex
{
   const CGRect rect = CGRectMake(0.f, 0.f, [ObjectShortcutView iconWidth], [ObjectShortcutView iconHeight]);
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
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   CGContextFillRect(ctx, rect);
   //Set context and paint pad's contents.
   ROOT::iOS::Pad *pad = fileContainer->GetPadAttached(objIndex);
   pad->cd();
   pad->SetViewWH(rect.size.width, rect.size.height);
   pad->SetContext(ctx);
   pad->PaintThumbnail();
   
   UIImage *thumbnailImage = UIGraphicsGetImageFromCurrentImageContext();//autoreleased UIImage.
   UIGraphicsEndImageContext();
       
   ObjectShortcutView * const shortcut = [[ObjectShortcutView alloc] initWithFrame : [ObjectShortcutView defaultRect]
                                          controller : self forObjectAtIndex : objIndex withThumbnail : thumbnailImage];
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
   ObjectShortcutView * const shortcut = [[ObjectShortcutView alloc] initWithFrame : [ObjectShortcutView defaultRect]
                                          controller : self forFolderAtIndex : index];
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
- (void) activateForFile : (ROOT::iOS::Browser::FileContainer *) container
{
   assert(container != nullptr && "activateForFile:, parameter 'container' is null");

   fileContainer = container;
}

//____________________________________________________________________________________________________
- (void) startSlideshow
{
   if (animating)
      return;

   assert(self.storyboard != nil && "startSlideshow, self.storyboard is nil");
   assert(fileContainer != nullptr && "startSlideshow, fileContainer is null");

   SlideshowViewController * const slideshowController = [self.storyboard instantiateViewControllerWithIdentifier:ROOT::iOS::Browser::SlideshowViewControllerID];
   [slideshowController setFileContainer : fileContainer];
   [self.navigationController pushViewController : slideshowController animated : YES];
}

//____________________________________________________________________________________________________
- (void) doTest
{
   const unsigned testIndex = 1 + rand() % (fileContainer->GetNumberOfObjects() - 1);
   ObjectViewController *objectController = [[ObjectViewController alloc] initWithNibName : @"ROOTObjectController" bundle : nil];
   [objectController setNavigationForObjectWithIndex : testIndex fromContainer : fileContainer];
   [self.navigationController pushViewController : objectController animated : YES];
}

//____________________________________________________________________________________________________
- (void) selectObjectFromFile : (ObjectShortcutView *) shortcut
{
   if (animating)
      return;

   assert(shortcut != nil && "selectObjectFromFile:, parameter shortcut is nil");
   assert(fileContainer != nullptr && "selectObjectFromFile:, fileContainer is null");
   assert(self.storyboard != nil && "selectObjectFromFile:, self.storyboard is nil");

   if (shortcut.isDirectory) {
      //Create another FileContentController and push it on stack.
      UIViewController * const c = (UIViewController *)[self.storyboard instantiateViewControllerWithIdentifier : ROOT::iOS::Browser::FileContentViewControllerID];
      assert([c isKindOfClass : [FileContentViewController class]] && "file content controller has a wrong type");
      FileContentViewController * const contentController = (FileContentViewController *)c;
      [contentController activateForFile : fileContainer->GetDirectory(shortcut.objectIndex)];
      [self.navigationController pushViewController : contentController animated : YES];
   } else {
      UIViewController * const c = (UIViewController *)[self.storyboard instantiateViewControllerWithIdentifier : ROOT::iOS::Browser::ObjectViewControllerID];
      assert([c isKindOfClass : [ObjectViewController class]] &&
             "object view controller has a wrong type");
      ObjectViewController * const objectController = (ObjectViewController *)c;
      [objectController setNavigationForObjectWithIndex : shortcut.objectIndex fromContainer : fileContainer];
      [self.navigationController pushViewController : objectController animated : YES];
   }
}

#pragma mark - Search delegate.

//____________________________________________________________________________________________________
- (void) searchBarTextDidBeginEditing : (UISearchBar *) aSearchBar
{
#pragma unused(aSearchBar)

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
- (void) searchBarTextDidEndEditing : (UISearchBar *) aSearchBar
{
   if (searchPopover) {
      [searchPopover dismissPopoverAnimated : YES];
      searchPopover = nil;
   }  

   [aSearchBar resignFirstResponder];
}

//____________________________________________________________________________________________________
- (void) searchBar : (UISearchBar *) searchBar textDidChange : (NSString *) searchText
{
#pragma unused(searchBar)
   // When the search string changes, filter the recents list accordingly.
   [searchController filterResultsUsingString : searchText];
}

//____________________________________________________________________________________________________
- (void) searchBarSearchButtonClicked : (UISearchBar *) aSearchBar
{
#pragma unused(aSearchBar)

   [searchPopover dismissPopoverAnimated : YES];
   [searchBar resignFirstResponder];
}

#pragma mark - Popover controller delegate.

//____________________________________________________________________________________________________
- (void) popoverControllerDidDismissPopover : (UIPopoverController *) popoverController
{
#pragma unused(popoverController)
   [searchBar resignFirstResponder];
}

#pragma mark - Search delegate.

//____________________________________________________________________________________________________
- (void) searchController : (SearchViewController *) controller didSelectKey : (FileContainerElement *) key
{
#pragma unused(controller)

   assert(key != nil && "searcheController:didSelectKey:, parameter 'key' is nil");
   assert(key.elementIndex < fileContainer->GetNumberOfDescriptors() &&
          "searcheController:didSelectKey:, key.elementIndex is out of bounds");

   [searchPopover dismissPopoverAnimated : YES];
   searchPopover = nil;
   [searchBar resignFirstResponder];
   
   const auto &descriptor = fileContainer->GetElementDescriptor(key.elementIndex);
   if (descriptor.fOwner == fileContainer) {
      descriptor.fIsDir ? [self highlightDirectory : descriptor.fIndex] : [self highlightObject : descriptor.fIndex];
   } else {
      //Create another FileContentController and push it on stack.
      assert(self.storyboard != nil && "searcheController:didSelectKey:, self.storyboard is nil");
      UIViewController * const c = (UIViewController *)[self.storyboard instantiateViewControllerWithIdentifier : ROOT::iOS::Browser::FileContentViewControllerID];
      assert([c isKindOfClass : [FileContentViewController class]] &&
             "searcheController:didSelectKey, file content controller has a wrong type");
      FileContentViewController * const contentController = (FileContentViewController *)c;
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
- (void) animateShortcut : (ObjectShortcutView *) sh
{
   assert(sh != nil && "animateShortcut:, parameter 'sh' is nil");
   
   const CGRect oldFrame = sh.frame;
   const CGAffineTransform originalTransform = sh.transform;
   const CGAffineTransform newTransform = CGAffineTransformScale(originalTransform, 1.2f, 1.2f);

   sh.transform = newTransform;
   sh.spot.alpha = 0.8f;
   
   animating = YES;

   [UIView beginAnimations : @"show_object" context : nil];
   [UIView setAnimationDuration : 0.5f];
   [UIView setAnimationCurve : UIViewAnimationCurveLinear];
   [UIView setAnimationTransition : UIViewAnimationTransitionNone forView : sh cache : YES];
   [UIView setAnimationDelegate : self];
   [UIView setAnimationDidStopSelector : @selector(animationDidStop:finished:context:)];
   sh.transform = originalTransform;
   sh.spot.alpha = 0.f;
   sh.frame = oldFrame;
   [UIView commitAnimations];
}

//____________________________________________________________________________________________________
- (void) animationDidStop : (NSString *) animationID finished : (NSNumber *) finished context : (void *) context
{
#pragma unused(animationID, context)
   if ([finished boolValue])
      animating = NO;
}

//____________________________________________________________________________________________________
- (void) highlightDirectory : (unsigned) tag
{
   if (animating)
      return;

   for (ObjectShortcutView *sh in objectShortcuts) {
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
- (void) highlightObject : (unsigned) tag
{
   if (animating)
      return;

   for (ObjectShortcutView *sh in objectShortcuts) {
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
