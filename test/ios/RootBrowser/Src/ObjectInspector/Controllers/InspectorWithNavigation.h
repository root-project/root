#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface InspectorWithNavigation : UINavigationController <ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) obj;
- (NSString *) getComponentName;
- (void) resetInspector;

@end
