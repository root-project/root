#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface AxisTicksInspector : UIViewController <ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) object;


- (IBAction) plusTick : (UIButton *) sender;
- (IBAction) minusTick :(UIButton *) sender;
- (IBAction) ticksNegPosPressed;

@end
