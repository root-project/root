#import "ObjectInspectorComponent.h"

//
//The name is not really good: this inspector's view contains:
//range slider to change X-axis range; text field for H1's title;
//switch control (show or hide markers).
//I think the name is a legacy (at the beginning inspector was quite different).
//It's one of two tabs in the 'Hist attributes' inspector.
//

@interface H1BinsInspector : UIViewController <ObjectInspectorComponent>

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

@end
