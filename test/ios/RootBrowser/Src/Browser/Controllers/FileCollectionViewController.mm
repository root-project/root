#import <cassert>

#import "FileCollectionViewController.h"
#import "FileContentViewController.h"
#import "FileShortcutView.h"
#import "Shortcuts.h"
#import "Constants.h"

//C++ imports.
#import "FileUtils.h"


@implementation FileCollectionViewController {
   __weak IBOutlet UIScrollView *scrollView;
   __weak IBOutlet UIView *fileOpenView;
   __weak IBOutlet UITextField *fileNameField;
   
   NSMutableArray *fileContainers;
   BOOL viewDidAppear;
   __weak FileShortcutView *fileToDelete;
   
   BOOL locked;
}

//____________________________________________________________________________________________________
- (instancetype) initWithCoder : (NSCoder *) aDecoder
{
   if (self = [super initWithCoder : aDecoder]) {
      fileContainers = [[NSMutableArray alloc] init];
      viewDidAppear = NO;
      locked = NO;
   }
   
   return self;
}

#pragma mark - View lifecycle

//____________________________________________________________________________________________________
- (void) viewDidLoad
{
   [super viewDidLoad];

   //Setup additional views here.
   self.navigationItem.title = @"ROOT files";
   self.navigationItem.backBarButtonItem = [[UIBarButtonItem alloc] initWithTitle : @"Back to ROOT files"
                                             style : UIBarButtonItemStylePlain target : nil action : nil];
   self.navigationItem.leftBarButtonItem = [[UIBarButtonItem alloc] initWithTitle : @"Open file" style : UIBarButtonItemStylePlain target : self action : @selector(showFileOpenView)];

   scrollView.bounces = NO;
   
   [self.view bringSubviewToFront : fileOpenView];//it's still hidden.

   fileNameField.clearButtonMode = UITextFieldViewModeAlways;
   
   UITapGestureRecognizer * const tap = [[UITapGestureRecognizer alloc] initWithTarget : self action : @selector(hideFileOpenView)];
   //Usual Apple's fu..up with gestures.
   tap.delegate = self;
   [self.view addGestureRecognizer : tap];
}

//____________________________________________________________________________________________________
- (void) viewWillAppear : (BOOL)animated
{
   [super viewWillAppear : animated];

   [fileNameField resignFirstResponder];//? TODO: check why do I have this here at all.
}

//____________________________________________________________________________________________________
- (void) viewDidLayoutSubviews
{
   //I only have to set correct positions for the shortcuts.
   [self correctFramesForOrientation : self.interfaceOrientation];
}

//____________________________________________________________________________________________________
- (void) viewDidAppear : (BOOL) animated
{
   [super viewDidAppear : animated];

   if (!viewDidAppear) {
      //The first time this method is called, add the 'demos.root'.
      if (fileContainers.count) {
         for (UIView *shortcut in fileContainers)
            if (!shortcut.superview)
               [scrollView addSubview : shortcut];

         [self correctFramesForOrientation : self.interfaceOrientation];
      }
      
      viewDidAppear = YES;
   }
}

#pragma mark - View's geometry, interface orientation, etc.

//____________________________________________________________________________________________________
- (void) placeFileShortcuts
{
   using ROOT::iOS::Browser::PlaceShortcutsInScrollView;
   
   if ([scrollView.subviews count])
      //25.f - is an additional 'pad' space between shortcuts.
      PlaceShortcutsInScrollView(fileContainers, scrollView, CGSizeMake([FileShortcutView iconWidth], [FileShortcutView iconHeight]), 25.f);
}

//____________________________________________________________________________________________________
- (void) correctFramesForOrientation : (UIInterfaceOrientation) orientation
{
#pragma unused(orientation)

   //This is the legacy code: before I was resetting views geometry manually, now no
   //need in this nightmare anymore, just place shortcuts in correct places.

   [self placeFileShortcuts];
}


//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

   return YES;
}

//____________________________________________________________________________________________________
- (void) willAnimateRotationToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation duration : (NSTimeInterval) duration
{
#pragma unused(duration)

   [self correctFramesForOrientation : interfaceOrientation];
}

#pragma mark - File shortcuts.

//____________________________________________________________________________________________________
- (void) addRootFile : (NSString *) fileName
{
   //Open the file and read its contents.
   assert(fileName != nil && "addRootFile:, parameter 'fileName' is nil");
   
   using namespace ROOT::iOS::Browser;
   
   FileContainer * const fileContainer = FileContainer::CreateFileContainer([fileName cStringUsingEncoding : [NSString defaultCStringEncoding]]);

   if (!fileContainer) {
      UIAlertView *alert = [[UIAlertView alloc] initWithTitle : @"File Open Error:"
                                                message : [NSString stringWithFormat:@"Could not open %@", fileName]
                                                delegate : nil
                                                cancelButtonTitle : @"Close"
                                                otherButtonTitles : nil];
      [alert show];
      return;
   }

   const CGRect shortcutFrame = CGRectMake(0.f, 0.f, [FileShortcutView iconWidth], [FileShortcutView iconHeight]);
   FileShortcutView * const newShortcut = [[FileShortcutView alloc] initWithFrame : shortcutFrame controller : self fileContainer : fileContainer];
   if (newShortcut) {
      [fileContainers addObject : newShortcut];
      
      if ([self isViewLoaded]) {
         [scrollView addSubview : newShortcut];
         [self placeFileShortcuts];
      }//else we'll do it later.
   }  else
      FileContainer::DeleteFileContainer(fileContainer);
}

//____________________________________________________________________________________________________
- (void) fileWasSelected : (FileShortcutView *) shortcut
{
   assert(shortcut != nil && "fileWasSelected:, parameter 'shortcut' is nil");

   if (locked || fileNameField.isFirstResponder)
      return;

   assert(self.storyboard != nil && "fileWasSelected:, self.storyboard is nil");

   UIViewController * const c = (UIViewController *)[self.storyboard instantiateViewControllerWithIdentifier : ROOT::iOS::Browser::FileContentViewControllerID];
   assert([c isKindOfClass : [FileContentViewController class]] &&
          "fileWasSelected:, file content view controller has a wrong type");

   FileContentViewController * const contentController = (FileContentViewController *)c;
   [contentController activateForFile : [shortcut getFileContainer]];
   [self.navigationController pushViewController : contentController animated : YES];
}

//____________________________________________________________________________________________________
- (void) tryToDelete : (FileShortcutView *) shortcut
{
   assert(shortcut != nil && "tryToDelete:, parameter 'shortcut' is nil");
   
   if (locked || fileNameField.isFirstResponder)
      return;

   NSString * const message = [NSString stringWithFormat : @"Do you really want to close %@?", shortcut.fileName];
   UIActionSheet * const dialog = [[UIActionSheet alloc] initWithTitle : message delegate : self
                                   cancelButtonTitle : @"Cancel" destructiveButtonTitle : @"Yes" otherButtonTitles : nil];
   fileToDelete = shortcut;
   [dialog showInView : self.view];
}

#pragma mark - Action sheet delegate, delete the file.

//____________________________________________________________________________________________________
- (void) actionSheet : (UIActionSheet *) actionSheet didDismissWithButtonIndex : (NSInteger) buttonIndex
{
#pragma unused(actionSheet)

   if (!buttonIndex) {
      [fileContainers removeObject:fileToDelete];
      [fileToDelete removeFromSuperview];
      [self correctFramesForOrientation : self.interfaceOrientation];
   }
}

#pragma mark - File open operations + text filed handling.

//____________________________________________________________________________________________________
- (void) animateFileOpenView
{
   assert(locked == NO && "animateFileOpenView, called while an animation is still active");
   //Do animation.
   // First create a CATransition object to describe the transition
   CATransition * const transition = [CATransition animation];
   transition.duration = 0.15;
   // using the ease in/out timing function
   transition.timingFunction = [CAMediaTimingFunction functionWithName : kCAMediaTimingFunctionEaseInEaseOut];
   // Now to set the type of transition.
   transition.type = kCATransitionPush;
   
   if (!fileOpenView.hidden)
      transition.subtype = kCATransitionFromBottom;
   else
      transition.subtype = kCATransitionFromTop;
   transition.delegate = self;
   // Next add it to the containerView's layer. This will perform the transition based on how we change its contents.
   [fileOpenView.layer addAnimation : transition forKey : nil];
}

//____________________________________________________________________________________________________
- (void) showFileOpenView
{
   if (locked)//We're animating this view already.
      return;

   fileOpenView.hidden = !fileOpenView.hidden;
   [self animateFileOpenView];

   if (!fileOpenView.hidden)
      [fileNameField becomeFirstResponder];
   else
      [fileNameField resignFirstResponder];
}

//____________________________________________________________________________________________________
- (IBAction) textFieldDidEndOnExit : (id) sender
{
#pragma unused(sender)
   NSString * const filePath = fileNameField.text;
   if (filePath)
      [self addRootFile : filePath];
}

//____________________________________________________________________________________________________
- (IBAction) textFieldEditingDidEnd : (id) sender
{
#pragma unused(sender)
   [sender resignFirstResponder];
   fileOpenView.hidden = YES;
   [self animateFileOpenView];
}

//____________________________________________________________________________________________________
- (void) hideFileOpenView
{
   [fileNameField resignFirstResponder];
   fileOpenView.hidden = YES;
   [self animateFileOpenView];
}

#pragma mark - CAAnimationDelegate.

//____________________________________________________________________________________________________
- (void) animationDidStart : (CAAnimation *) anim
{
#pragma unused(anim)
   locked = YES;
}

//____________________________________________________________________________________________________
- (void) animationDidStop : (CAAnimation *) anim finished : (BOOL) flag
{
#pragma unused(anim)
   if (flag)
      locked = NO;
}

#pragma mark - Standard f..p with gestures.

//____________________________________________________________________________________________________
- (BOOL) gestureRecognizer : (UIGestureRecognizer *) gestureRecognizer shouldReceiveTouch : (UITouch *) touch
{
#pragma unused(gestureRecognizer)
   if (locked)
      return NO;

   if ([touch.view isKindOfClass : [FileShortcutView class]])
      return NO;
   
   return YES;
}

@end
