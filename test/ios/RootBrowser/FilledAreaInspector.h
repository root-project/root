#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"

@interface FilledAreaInspector : UIViewController <HorizontalPickerDelegate, ObjectInspectorComponent>

- (void) setROOTObjectController : (ROOTObjectController *) p;
- (void) setROOTObject : (TObject*) obj;
- (NSString *) getComponentName;

@end
