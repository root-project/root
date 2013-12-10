#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

//
//We have a "3D" navigation inside the Object Inspector:
//shrinking/extending views + tabs in some inspectors +
//a navigation controllers in some inspectors.
//InspectorWithNavigation is a UINavigationController's subcluss supporting ObjectInspectorComponent's
//protocol.

@interface InspectorWithNavigation : UINavigationController <ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) obj;
- (NSString *) getComponentName;
- (void) resetInspector;

@end
