#import "ObjectInspectorComponent.h"

@class ROOTObjectController;


class TObject;
class TAxis;

@interface AxisLabelsInspector : UIViewController <ObjectInspectorComponent>

+ (CGRect) inspectorFrame;

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

- (IBAction) showLabelFontInspector;

- (IBAction) plusBtn : (UIButton *)sender;
- (IBAction) minusBtn : (UIButton *)sender;
- (IBAction) noExpPressed;

@end
