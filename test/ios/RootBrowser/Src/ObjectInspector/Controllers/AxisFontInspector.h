#import "ObjectInspectorComponent.h"

//A simple view/controller.
//AxisFontInspector's view contains a picker with font names and a 'back' button.
//New font is set for either axis title or axis labels.

@interface AxisFontInspector : UIViewController <UIPickerViewDelegate, UIPickerViewDataSource, ObjectInspectorComponent>

//For for axis title or labels.
- (instancetype) initWithNibName : (NSString *) nibName isTitle : (BOOL) isTitle;

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

- (IBAction) back;

@end
