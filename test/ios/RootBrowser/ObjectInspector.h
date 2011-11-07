#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class EditorView;

@interface ObjectInspector : UIViewController <ObjectInspectorComponent> 

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;
- (void) resetInspector;

- (EditorView *) getEditorView;

@end
