#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"


@class HorizontalPickerView;
@class LineWidthPicker;

//Line inspector is a composition of two sub-inspectors: line color and width inspector + 
//line style inspector.

@interface LineInspector : UIViewController <ObjectInspectorComponent, HorizontalPickerDelegate>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;
- (NSString *) getComponentName;

- (void) item : (unsigned int)item wasSelectedInPicker : (HorizontalPickerView *) picker;

- (IBAction) decLineWidth;
- (IBAction) incLineWidth;

@end
