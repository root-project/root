#import <UIKit/UIKit.h>

#import "TAttMarker.h"

@interface MarkerStyleCell : UIView {
@private
   EMarkerStyle markerStyle;
}

- (id) initWithFrame : (CGRect)frame andMarkerStyle : (EMarkerStyle)style;

@end
