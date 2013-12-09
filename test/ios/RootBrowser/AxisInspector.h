#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface AxisInspector : UIViewController <ObjectInspectorComponent, UITabBarDelegate> {
@private
   __weak IBOutlet UITabBar *tabBar;
}

+ (CGRect) inspectorFrame;

- (void) setROOTObjectController : (ObjectViewController *)c;
- (void) setROOTObject : (TObject *)o;
- (NSString *) getComponentName;
- (void) resetInspector;



@end
