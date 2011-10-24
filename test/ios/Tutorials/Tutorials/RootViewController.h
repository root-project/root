//
//  RootViewController.h
//  Tutorials
//
//  Created by Timur Pocheptsov on 7/7/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <UIKit/UIKit.h>

#import "DemoBase.h"

@class DetailViewController;
@class NSMutableArray;
@class NSTimer;

enum {
   nROOTDemos = 6
};

@interface RootViewController : UITableViewController {
   NSMutableArray *tutorialNames;
   NSMutableArray *tutorialIcons;
   
   NSTimer *animationTimer;
   
   ROOT::iOS::Demos::DemoBase *demos[nROOTDemos];
}
		
@property (nonatomic, retain) IBOutlet DetailViewController *detailViewController;

@end
