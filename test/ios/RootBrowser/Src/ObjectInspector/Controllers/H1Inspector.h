#import "ObjectInspectorComponent.h"

//
//Object inspector with nested object inspectors (inside tab bar).
//

@interface H1Inspector : UIViewController <ObjectInspectorComponent, UITabBarDelegate>

- (void) setObject : (TObject *) o;
- (void) setObjectController : (ObjectViewController *) c;
- (NSString *) getComponentName;
- (void) resetInspector;


@end
