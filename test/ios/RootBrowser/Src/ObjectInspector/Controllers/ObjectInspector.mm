#import <cassert>

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

namespace {
   enum {
      //Just indices.
      kAttLine = 0,
      kAttFill = 1,
      kAttPad = 2,
      kAttAxis  = 3,
      //Add the new one here.
      kAttMarker = 4,
      kAttH1 = 5,
      kNOfInspectors //
   };
}

@implementation ObjectInspector {
   UIViewController <ObjectInspectorComponent> *activeEditors[kNOfInspectors];
   UIViewController <ObjectInspectorComponent> *cachedEditors[kNOfInspectors];

   unsigned nActiveEditors;
   
   TObject *object;
   
   EditorView *editorView;
}


//____________________________________________________________________________________________________
- (void) initObjectInspectorView
{
   editorView = [[EditorView alloc] initWithFrame:CGRectMake(0.f, 0.f, [EditorView editorWidth], [EditorView editorHeight])];
   self.view = editorView;
}

//____________________________________________________________________________________________________
- (void) cacheEditors
{
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
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];

   if (self) {
      [self initObjectInspectorView];
      [self cacheEditors];
   }

   return self;
}

#pragma mark - Interface orientation.

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

	return YES;
}

//____________________________________________________________________________________________________
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");
   
   for (unsigned i = 0; i < nActiveEditors; ++i)
      [activeEditors[i] setObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) o
{
   assert(o != nullptr && "setObject:, parameter 'o' is null");

   if (o != object) {
      //Initialize.
      object = o;
      
      [self setTitle];
      [self setActiveEditors];
   
      for (unsigned i = 0; i < nActiveEditors; ++i)
         [activeEditors[i] setObject : o];
      
      [editorView removeAllEditors];

      for (unsigned i = 0; i < nActiveEditors; ++i)
         [editorView addSubEditor : activeEditors[i].view withName : [activeEditors[i] getComponentName]];
   }
}

//____________________________________________________________________________________________________
- (void) setTitle
{
   assert(object != nullptr && "setTitle, object is null");

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
   nActiveEditors = 0;

   if (dynamic_cast<TAttLine *>(object) && !dynamic_cast<TAttPad *>(object))
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
