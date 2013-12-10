#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

//
//Title inspect is one of tabs inside an AxisInspector's tab view.
//It has several controls and is a part of a nested navigation stack.
//

@interface AxisTitleInspector : UIViewController <ObjectInspectorComponent>

+ (CGRect) inspectorFrame;

- (void) setObjectController : (ObjectViewController *)c;
- (void) setObject : (TObject *)o;

@end
