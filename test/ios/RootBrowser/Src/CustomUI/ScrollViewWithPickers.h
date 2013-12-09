#import <UIKit/UIKit.h>

//This is small class, inheriting UIScrollView.
//The only reason to exist at all - inspector's
//window has a scroll-view, which can include
//UIPickerView, and UIPickerView is not so easy
//to rotate in a scroll-view.

@interface ScrollViewWithPickers : UIScrollView

@end
