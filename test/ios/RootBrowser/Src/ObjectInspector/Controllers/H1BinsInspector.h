#import "ObjectInspectorComponent.h"

@interface H1BinsInspector : UIViewController <ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

@end
