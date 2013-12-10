#import "ObjectInspectorComponent.h"

//
//A very simple object inspector with a picker view to select an error type.
//One of two tabs inside an H1Inspector's view.
//

@interface H1ErrorsInspector : UIViewController <UIPickerViewDelegate, UIPickerViewDataSource, ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

@end
