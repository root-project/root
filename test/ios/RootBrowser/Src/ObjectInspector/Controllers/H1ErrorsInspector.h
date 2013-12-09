#import "ObjectInspectorComponent.h"

@interface H1ErrorsInspector : UIViewController <UIPickerViewDelegate, UIPickerViewDataSource, ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

@end
