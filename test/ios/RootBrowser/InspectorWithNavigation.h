#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface InspectorWithNavigation : UINavigationController <ObjectInspectorComponent>

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)obj;
- (NSString *) getComponentName;
- (void) resetInspector;

@end
