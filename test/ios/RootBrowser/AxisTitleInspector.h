#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface AxisTitleInspector : UIViewController <ObjectInspectorComponent>

+ (CGRect) inspectorFrame;

- (void) setObjectController : (ObjectViewController *)c;
- (void) setObject : (TObject *)o;

- (IBAction) showTitleFontInspector;
- (IBAction) textFieldDidEndOnExit : (id) sender;
- (IBAction) textFieldEditingDidEnd : (id) sender;
- (IBAction) centerTitle;
- (IBAction) rotateTitle;
- (IBAction) plusOffset;
- (IBAction) minusOffset;
- (IBAction) plusSize;
- (IBAction) minusSize;

@end
