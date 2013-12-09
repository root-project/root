//
//  Shortcuts.h
//  root_browser
//
//  Created by Timur Pocheptsov on 8/22/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {
namespace Browser {

//We assume, that 'shortcuts' (some views) are already subviews of a scrollview, I do not check it.
void PlaceShortcutsInScrollView(NSMutableArray *shortcuts, UIScrollView *scrollView, const CGSize &shortCutSize, CGFloat padSpace);

}
}
}