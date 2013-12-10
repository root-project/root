#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface PadTicksGridInspector : UIViewController <ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) obj;

@end
