#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"

@interface MarkerInspector : UIViewController <ObjectInspectorComponent, HorizontalPickerDelegate>

- (void) setObjectController : (ObjectViewController *)c;
- (void) setObject : (TObject *)o;
- (NSString *) getComponentName;

@end
