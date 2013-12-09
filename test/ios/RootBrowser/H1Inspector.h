#import "ObjectInspectorComponent.h"

@interface H1Inspector : UIViewController <ObjectInspectorComponent>

- (void) setObject : (TObject *) o;
- (void) setObjectController : (ObjectViewController *) c;
- (NSString *) getComponentName;
- (void) resetInspector;


@end
