#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class ROOTObjectController;

class TObject;

@interface InspectorWithNavigation : UINavigationController <ObjectInspectorComponent>

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)obj;
- (NSString *) getComponentName;
- (void) resetInspector;

@end
