#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"

@interface FilledAreaInspector : UIViewController <HorizontalPickerDelegate, ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) p;
- (void) setObject : (TObject*) obj;
- (NSString *) getComponentName;

@end
