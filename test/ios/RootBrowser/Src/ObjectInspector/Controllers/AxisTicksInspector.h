#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

//AxisTicksInspector: its view is nested inside an AxisInspector's view (as a tab).

@interface AxisTicksInspector : UIViewController <ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) object;

@end
