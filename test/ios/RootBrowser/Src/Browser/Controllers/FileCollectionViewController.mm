#import <cassert>

#import <QuartzCore/QuartzCore.h>

#import "FileCollectionViewController.h"
#import "FileContentViewController.h"
#import "FileShortcutView.h"
#import "Shortcuts.h"

//C++ imports.
#import "FileUtils.h"


@implementation FileCollectionViewController {
   __weak IBOutlet UIScrollView *scrollView;
   __weak IBOutlet UIView *fileOpenView;
   __weak IBOutlet UITextField *fileNameField;
   
   NSMutableArray *fileContainers;
   BOOL viewDidAppear;
   __weak FileShortcutView *fileToDelete;
}

//____________________________________________________________________________________________________
- (id) initWithCoder : (NSCoder *) aDecoder
{
   if (self = [super initWithCoder : aDecoder]) {
      fileContainers = [[NSMutableArray alloc] init];
      viewDidAppear = NO;
   }
   
   return self;
}

//____________________________________________________________________________________________________
- (id) initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
   
   if (self) {
      [self view];
      
      fileContainers = [[NSMutableArray alloc] init];
   
      self.navigationItem.title = @"ROOT files";
      self.navigationItem.backBarButtonItem = [[UIBarButtonItem alloc] initWithTitle : @"Back to ROOT files" style:UIBarButtonItemStylePlain target : nil action : nil];
      self.navigationItem.leftBarButtonItem = [[UIBarButtonItem alloc] initWithTitle : @"Open file" style:UIBarButtonItemStylePlain target : self action : @selector(showFileOpenView)];

      scrollView.bounces = NO;
      
      [self.view bringSubviewToFront : fileOpenView];
      
      fileNameField.clearButtonMode = UITextFieldViewModeAlways;
      
      UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(hideFileOpenView)];
      [self.view addGestureRecognizer : tap];
   }

   return self;
}

#pragma mark - View lifecycle

//____________________________________________________________________________________________________
- (void) placeFileShortcuts
{
   if ([scrollView.subviews count])
      [ShorcutUtil placeShortcuts : fileContainers inScrollView : scrollView withSize : CGSizeMake([FileShortcutView iconWidth], [FileShortcutView iconHeight]) andSpace : 25.f];
}

//____________________________________________________________________________________________________
- (void) correctFramesForOrientation : (UIInterfaceOrientation) orientation
{
   CGRect mainFrame;
   CGRect scrollFrame;
   CGRect fileViewFrame;

   if (UIInterfaceOrientationIsPortrait(orientation)) {
      mainFrame = CGRectMake(0.f, 0.f, 768.f, 1004.f);
      scrollFrame = CGRectMake(0.f, 44.f, 768.f, 960.f);
      fileViewFrame = CGRectMake(0.f, 44.f, 768.f, 120.f);
   } else {
      mainFrame = CGRectMake(0.f, 0.f, 1024.f, 748.f);
      scrollFrame = CGRectMake(0.f, 44.f, 1024.f, 704.f);   
      fileViewFrame = CGRectMake(0.f, 44.f, 1024.f, 120.f);
   }
   
   self.view.frame = mainFrame;
   scrollView.frame = scrollFrame;
   
   fileOpenView.frame = fileViewFrame;
   
   [self placeFileShortcuts];
}

//____________________________________________________________________________________________________
- (void) viewWillAppear : (BOOL)animated
{
   [self correctFramesForOrientation : self.interfaceOrientation];
   [fileNameField resignFirstResponder];
   //Check if I have to call [super viewWillAppear];
}

//____________________________________________________________________________________________________
- (void) viewDidLoad
{
   [super viewDidLoad];
   //
   //Setup additional views here.
   
   //
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

#pragma mark - View management.

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
   if (newShortcut) {//What if alloc returned nil?
      [fileContainers addObject : newShortcut];
      [scrollView addSubview : newShortcut];   
      [self placeFileShortcuts];
   }  else
      FileContainer::DeleteFileContainer(fileContainer);
}

//____________________________________________________________________________________________________
- (void) fileWasSelected : (FileShortcutView *) shortcut
{
   FileContentViewController *contentController = [[FileContentViewController alloc] initWithNibName : @"FileContentController" bundle : nil];
   [contentController activateForFile : [shortcut getFileContainer]];
   [self.navigationController pushViewController : contentController animated : YES];
}

//____________________________________________________________________________________________________
- (void) tryToDelete : (FileShortcutView *) shortcut
{
   NSString *message = [NSString stringWithFormat : @"Do you really want to close %@?", shortcut.fileName];
   UIActionSheet *dialog = [[UIActionSheet alloc] initWithTitle : message delegate : self cancelButtonTitle : @"Cancel" destructiveButtonTitle : @"Yes" otherButtonTitles : nil];
   fileToDelete = shortcut;
   [dialog showInView : self.view];
}

#pragma mark - Action sheet delegate, delete the file.

//____________________________________________________________________________________________________
- (void)actionSheet : (UIActionSheet *)actionSheet didDismissWithButtonIndex : (NSInteger)buttonIndex
{
   if (!buttonIndex) {
      [fileContainers removeObject:fileToDelete];
      [fileToDelete removeFromSuperview];
      [self correctFramesForOrientation : self.interfaceOrientation];
   }
}

#pragma mark - File open operations.

//____________________________________________________________________________________________________
- (void) animateFileOpenView
{
   //Do animation.
   // First create a CATransition object to describe the transition
   CATransition *transition = [CATransition animation];
   transition.duration = 0.15;
   // using the ease in/out timing function
   transition.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseInEaseOut];
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
   fileOpenView.hidden = !fileOpenView.hidden;
   //
   [self animateFileOpenView];

   if (!fileOpenView.hidden)
      [fileNameField becomeFirstResponder];
   else
      [fileNameField resignFirstResponder];
}

//____________________________________________________________________________________________________
- (IBAction) textFieldDidEndOnExit : (id) sender
{
   NSString *filePath = fileNameField.text;
   if (filePath) {//TODO - do I need this check?
      [self addRootFile : filePath];
   }
}

//____________________________________________________________________________________________________
- (IBAction) textFieldEditingDidEnd : (id) sender
{
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

@end
