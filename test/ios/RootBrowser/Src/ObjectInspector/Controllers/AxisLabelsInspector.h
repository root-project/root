#import "ObjectInspectorComponent.h"

//
//AxisLabelInspector, placed in a tab inside an AxisInspector's view.
//Has several controlls and is a part of a nested navigation stack -
//label font (AxisFontInspector) is pushed into the navigation stack.
//

@class ROOTObjectController;

class TObject;
class TAxis;

@interface AxisLabelsInspector : UIViewController <ObjectInspectorComponent>

+ (CGRect) inspectorFrame;

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

@end
