#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"

//
//Object inspector to work with a TAttMarker, has:
//two horizontal pickers (marker type and color) + marker size.
//

@interface MarkerInspector : UIViewController <ObjectInspectorComponent, HorizontalPickerDelegate>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;
- (NSString *) getComponentName;

@end
