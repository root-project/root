#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

//
//Object inspector (one of two tabs) for a pad's log scales.
//

@interface PadLogScaleInspector : UIViewController <ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

@end
