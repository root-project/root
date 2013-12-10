#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"

//ObjectInspector to work with a TAttFill object, view contains two horizontal pickers - color/pattern selectors.

@interface FilledAreaInspector : UIViewController <HorizontalPickerDelegate, ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) p;
- (void) setObject : (TObject*) obj;
- (NSString *) getComponentName;

@end
