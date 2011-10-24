#import "InspectorWithNavigation.h"
#import "FilledAreaInspector.h"
#import "ObjectInspector.h"
#import "MarkerInspector.h"
#import "AxisInspector.h"
#import "LineInspector.h"
#import "PadInspector.h"
#import "H1Inspector.h"
#import "EditorView.h"

//C++ (ROOT) imports.
#import "TAttMarker.h"
#import "TAttLine.h"
#import "TAttFill.h"
#import "TAttAxis.h"
#import "TAttPad.h"
#import "TObject.h"
#import "TClass.h"
#import "TH1.h"

@implementation ObjectInspector

//____________________________________________________________________________________________________
- (void) initObjectInspectorView
{
   editorView = [[EditorView alloc] initWithFrame:CGRectMake(0.f, 0.f, [EditorView editorWidth], [EditorView editorHeight])];
   self.view = editorView;
   [editorView release];
}

//____________________________________________________________________________________________________
- (void) cacheEditors
{
   using namespace ROOT_IOSObjectInspector;

   //TAttLine.
   cachedEditors[kAttLine] = [[LineInspector alloc] initWithNibName : @"LineInspector" bundle : nil];//lineInspector;   
   //TAttFill.
   cachedEditors[kAttFill] = [[FilledAreaInspector alloc] initWithNibName : @"FilledAreaInspector" bundle : nil];
   //TAttPad.
   cachedEditors[kAttPad] = [[PadInspector alloc] initWithNibName : @"PadInspector" bundle : nil];//padInspector;   
   //TAttAxis.
   cachedEditors[kAttAxis] = [[AxisInspector alloc] initWithNibName : @"AxisInspector" bundle : nil];
   //TAttMarker.
   cachedEditors[kAttMarker] = [[MarkerInspector alloc] initWithNibName: @"MarkerInspector" bundle : nil];
   //H1's inspector.
   cachedEditors[kAttH1] = [[H1Inspector alloc] initWithNibName : @"H1Inspector" bundle : nil];
}

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
   if (self) {
      [self initObjectInspectorView];
      [self cacheEditors];
   }
   return self;
}

//____________________________________________________________________________________________________
- (void) dealloc
{
   using ROOT_IOSObjectInspector::kNOfInspectors;
   for (unsigned i = 0; i < kNOfInspectors; ++i)
      [cachedEditors[i] release];

   [super dealloc];
}

//____________________________________________________________________________________________________
- (void)didReceiveMemoryWarning
{
   // Releases the view if it doesn't have a superview.
   [super didReceiveMemoryWarning];
   // Release any cached data, images, etc that aren't in use.
}

#pragma mark - View lifecycle

/*
// Implement loadView to create a view hierarchy programmatically, without using a nib.
- (void)loadView
{
}
*/

/*
// Implement viewDidLoad to do additional setup after loading the view, typically from a nib.
- (void)viewDidLoad
{
    [super viewDidLoad];
}
*/

//____________________________________________________________________________________________________
- (void)viewDidUnload
{
    [super viewDidUnload];
    // Release any retained subviews of the main view.
    // e.g. self.myOutlet = nil;
}

//____________________________________________________________________________________________________
- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
    // Return YES for supported orientations
	return YES;
}

//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *)c
{
   for (unsigned i = 0; i < nActiveEditors; ++i)
      [activeEditors[i] setROOTObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setTitle
{
   if (dynamic_cast<TAttPad *>(object)) {
      //This is special case, as soon as ROOT::iOS::Pad does not have
      //ClassDef, the IsA() will be for TVirtualPad, but I want to
      //see simply "Pad" as a title.
      [editorView setEditorTitle : "Pad"];
   } else {
      [editorView setEditorTitle : object->IsA()->GetName()];
   }
}

//____________________________________________________________________________________________________
- (void) setActiveEditors
{
   using namespace ROOT_IOSObjectInspector;

   nActiveEditors = 0;

   if (dynamic_cast<TAttLine *>(object))
      activeEditors[nActiveEditors++] = cachedEditors[kAttLine];
   
   if (dynamic_cast<TAttFill *>(object))
      activeEditors[nActiveEditors++] = cachedEditors[kAttFill];
   
   if (dynamic_cast<TAttPad *>(object))
      activeEditors[nActiveEditors++] = cachedEditors[kAttPad];
      
   if (dynamic_cast<TAttAxis *>(object))
      activeEditors[nActiveEditors++] = cachedEditors[kAttAxis];
   
   if (dynamic_cast<TAttMarker *>(object))
      activeEditors[nActiveEditors++] = cachedEditors[kAttMarker];

   if (dynamic_cast<TH1 *>(object))
      activeEditors[nActiveEditors++] = cachedEditors[kAttH1];
}

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)o
{
   if (o != object) {
      //Initialize.
      object = o;
      
      [self setTitle];
      [self setActiveEditors];
   
      for (unsigned i = 0; i < nActiveEditors; ++i)
         [activeEditors[i] setROOTObject : o];
      
      [editorView removeAllEditors];

      for (unsigned i = 0; i < nActiveEditors; ++i)
         [editorView addSubEditor : activeEditors[i].view withName : [activeEditors[i] getComponentName]];
   }
}

//____________________________________________________________________________________________________
- (void) resetInspector
{
   for (unsigned i = 0; i < nActiveEditors; ++i)
      if ([activeEditors[i] respondsToSelector : @selector(resetInspector)])
         [activeEditors[i] resetInspector];
}

//____________________________________________________________________________________________________
- (EditorView *) getEditorView
{
   return editorView;
}

@end
