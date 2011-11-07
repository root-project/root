#import <UIKit/UIKit.h>


namespace ROOT {
namespace iOS {

class FileContainer;

}
}

@interface FileShortcut : UIView

@property (nonatomic, retain) NSString *fileName;

+ (CGFloat) iconWidth;
+ (CGFloat) iconHeight;

- (id) initWithFrame : (CGRect)frame controller : (UIViewController *)controller fileContainer : (ROOT::iOS::FileContainer *)container;
- (ROOT::iOS::FileContainer *) getFileContainer;

@end
