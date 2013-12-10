#import <cassert>

#import "ObjectViewController.h"
#import "AxisLabelsInspector.h"
#import "AxisFontInspector.h"

#import "TObject.h"
#import "TAxis.h"

//It's mm file == C++, consts have internal linkage.
const float sizeStep = 0.01f;
const float minSize = 0.f;
const float maxSize = 1.f;

const float offsetStep = 0.001f;
const float minOffset = -1.f;
const float maxOffset = 1.f;

const float totalHeight = 400.f;
const float tabBarHeight = 49.f;
const CGRect componentFrame = CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);

@implementation AxisLabelsInspector {
   __weak IBOutlet UIButton *plusSize;
   __weak IBOutlet UIButton *minusSize;
   __weak IBOutlet UILabel *sizeLabel;
   
   __weak IBOutlet UIButton *plusOffset;
   __weak IBOutlet UIButton *minusOffset;
   __weak IBOutlet UILabel *offsetLabel;
   
   __weak IBOutlet UISwitch *noExp;

   __weak ObjectViewController *controller;
   TAxis *object;
}

//____________________________________________________________________________________________________
+ (CGRect) inspectorFrame
{
   return componentFrame;
}

//____________________________________________________________________________________________________
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   if (self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil]) {
      //Force loading self.view and subviews.
      [self view];
   }

   return self;
}

#pragma mark - Interface orientation.

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

	return YES;
}

#pragma mark - ObjectInspectorComponent.

//____________________________________________________________________________________________________
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");
   controller = c;
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) o
{
   assert(o != nullptr && "setObject:, parameter 'o' is null");

   object = dynamic_cast<TAxis *>(o);
   //The result of a cast is checked one level up.
   
   sizeLabel.text = [NSString stringWithFormat : @"%.2f", object->GetLabelSize()];
   offsetLabel.text = [NSString stringWithFormat : @"%.3f", object->GetLabelOffset()];
   
   noExp.on = object->GetNoExponent();
}

#pragma mark - UI interactions.

//____________________________________________________________________________________________________
- (IBAction) showLabelFontInspector
{
   AxisFontInspector * const fontInspector = [[AxisFontInspector alloc] initWithNibName : @"AxisFontInspector" isTitle : NO];

   [fontInspector setObjectController : controller];
   [fontInspector setObject : object];
   
   [self.navigationController pushViewController : fontInspector animated : YES];
}

//____________________________________________________________________________________________________
- (IBAction) plusBtn : (UIButton *) sender
{
   assert(object != nullptr && "plusBtn:, object is null");

   if (sender == plusSize) {
      if (object->GetLabelSize() + sizeStep > maxSize)
         return;
      
      sizeLabel.text = [NSString stringWithFormat : @"%.2f", object->GetLabelSize() + sizeStep];
      object->SetLabelSize(object->GetLabelSize() + sizeStep);
   } else if (sender == plusOffset) {
      if (object->GetLabelOffset() + offsetStep > maxOffset)
         return;
      
      offsetLabel.text = [NSString stringWithFormat : @"%.3f", object->GetLabelOffset() + offsetStep];
      object->SetLabelOffset(object->GetLabelOffset() + offsetStep);
   }
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) minusBtn : (UIButton *) sender
{
   assert(object != nullptr && "minusBtn:, object is null");

   if (sender == minusSize) {
      if (object->GetLabelSize() - sizeStep < minSize)
         return;
      
      sizeLabel.text = [NSString stringWithFormat : @"%.2f", object->GetLabelSize() - sizeStep];
      object->SetLabelSize(object->GetLabelSize() - sizeStep);
   } else if (sender == minusOffset) {
      if (object->GetLabelOffset() - offsetStep < minOffset)
         return;
      
      offsetLabel.text = [NSString stringWithFormat : @"%.3f", object->GetLabelOffset() - offsetStep];
      object->SetLabelOffset(object->GetLabelOffset() - offsetStep);
   }
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) noExpPressed
{
   assert(object != nullptr && "noExpPressed, object is null");

   object->SetNoExponent(noExp.on);
   [controller objectWasModifiedUpdateSelection : NO];
}

@end
