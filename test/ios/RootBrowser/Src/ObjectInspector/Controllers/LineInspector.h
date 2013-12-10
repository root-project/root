#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"


@class HorizontalPickerView;
@class LineWidthPicker;

//
//LineInspector: object inspector for a TAttLine, contains 2 horizontal pickers (color and dash style) +
//line width control.
//

@interface LineInspector : UIViewController <ObjectInspectorComponent, HorizontalPickerDelegate>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;
- (NSString *) getComponentName;

- (void) item : (unsigned int)item wasSelectedInPicker : (HorizontalPickerView *) picker;

@end
