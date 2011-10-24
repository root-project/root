#import <UIKit/UIKit.h>


namespace ROOT {
namespace iOS {

class FileContainer;

}
}

class TObject;

@interface FileShortcut : UIView {
@private
   UIViewController *controller;

   NSString *fileName;
   NSString *filePath;

   UIImage *filePictogram;
   UIImage *backgroundImage;
   
   ROOT::iOS::FileContainer *fileContainer;
}

@property (nonatomic, retain) NSString *fileName;
@property (nonatomic, retain) NSString *filePath;
@property (nonatomic, retain) NSString *errorMessage;

+ (CGFloat) iconWidth;
+ (CGFloat) iconHeight;

- (id) initWithFrame : (CGRect)frame controller : (UIViewController *)c filePath : (NSString *) path;

- (ROOT::iOS::FileContainer *) getFileContainer;

@end
