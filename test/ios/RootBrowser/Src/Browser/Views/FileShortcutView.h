#import <UIKit/UIKit.h>


namespace ROOT {
namespace iOS {
namespace Browser {

class FileContainer;

}
}
}

@interface FileShortcutView : UIView

@property (nonatomic, readonly) NSString *fileName;

+ (CGFloat) iconWidth;
+ (CGFloat) iconHeight;

- (instancetype) initWithFrame : (CGRect) frame controller : (UIViewController *) controller fileContainer : (ROOT::iOS::Browser::FileContainer *) container;
- (ROOT::iOS::Browser::FileContainer *) getFileContainer;

@end
