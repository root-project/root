#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface PadLogScaleInspector : UIViewController <ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *)c;
- (void) setObject : (TObject *)o;

- (IBAction) logActivated : (UISwitch *) log;

@end
