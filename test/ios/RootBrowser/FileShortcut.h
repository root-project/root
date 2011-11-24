#import <UIKit/UIKit.h>


namespace ROOT {
namespace iOS {
namespace Browser {

class FileContainer;

}
}
}

@interface FileShortcut : UIView

@property (nonatomic, retain) NSString *fileName;

+ (CGFloat) iconWidth;
+ (CGFloat) iconHeight;

- (id) initWithFrame : (CGRect)frame controller : (UIViewController *)controller fileContainer : (ROOT::iOS::Browser::FileContainer *)container;
- (ROOT::iOS::Browser::FileContainer *) getFileContainer;

@end
