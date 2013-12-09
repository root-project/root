#import <UIKit/UIKit.h>

#import "HorizontalPickerDelegate.h"

@interface HorizontalPickerView : UIView <UIScrollViewDelegate>

@property (nonatomic, weak) id<HorizontalPickerDelegate> pickerDelegate;

- (void) addItems : (NSMutableArray *)items;
- (void) setSelectedItem : (unsigned) item;

@end
