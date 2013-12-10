#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class EditorView;

@interface ObjectInspector : UIViewController <ObjectInspectorComponent> 

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;
- (void) resetInspector;

- (EditorView *) getEditorView;

@end
