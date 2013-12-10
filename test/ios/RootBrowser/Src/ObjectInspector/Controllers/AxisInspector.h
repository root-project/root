#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

//
//AxisInspector contains a tab-bar interface with nested sub-inspectors:
//AxisTicksInspector, AxisTitleInspector, AxisLabelsInspector.
//

@interface AxisInspector : UIViewController <ObjectInspectorComponent, UITabBarDelegate>

+ (CGRect) inspectorFrame;

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

- (NSString *) getComponentName;
- (void) resetInspector;



@end
