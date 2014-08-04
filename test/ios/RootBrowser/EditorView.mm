#import <math.h>

#import <CoreGraphics/CGContext.h>
#import <Availability.h>

#import "ScrollViewWithPickers.h"
#import "EditorPlateView.h"
#import "EditorView.h"


//Hoho! As soon as I use Objective-C++, I can use namespaces! "Yeaaahhh, that's good!" (c) Duke Nukem.

namespace {

enum {
   evMaxComponents = 5,
   evMaxStates = 1 << evMaxComponents
};

}

@implementation EditorView {
   UILabel *editorTitle;

   ScrollViewWithPickers *scrollView;

   CGFloat plateYs[evMaxStates * evMaxComponents];
   CGFloat viewYs[evMaxStates * evMaxComponents];

   UIView *plates[evMaxComponents];
   UIView *views[evMaxComponents];
   UIView *containers[evMaxComponents];

   unsigned nStates;
   unsigned nEditors;
   unsigned currentState;

   int newOpened;

   BOOL animation;
}

//____________________________________________________________________________________________________
+ (CGFloat) editorAlpha
{
   return 0.85f;
}

//____________________________________________________________________________________________________
+ (CGFloat) editorWidth
{
   return 270.f;
}

//____________________________________________________________________________________________________
+ (CGFloat) editorHeight
{
   return 650.f;
}

//____________________________________________________________________________________________________
+ (CGFloat) scrollWidth
{
   return [EditorView editorWidth] - 20.f;
}

//____________________________________________________________________________________________________
+ (CGFloat) scrollHeight
{
   return [EditorView editorHeight] - 20.f;
}

//____________________________________________________________________________________________________
+ (CGFloat) ncWidth
{
   return 20.f;
}

//____________________________________________________________________________________________________
+ (CGFloat) ncHeight
{
   return 20.f;
}

//____________________________________________________________________________________________________
- (id)initWithFrame : (CGRect)frame
{
   self = [super initWithFrame : frame];

   if (self) {
      //Scroll view is a container for all sub-editors.
      //It's completely transparent.
      const CGRect titleRect = CGRectMake(10.f, 10.f, 250.f, 35.f);
      editorTitle = [[UILabel alloc] initWithFrame : titleRect];

#ifdef __IPHONE_6_0
      editorTitle.textAlignment = NSTextAlignmentCenter;
#else
      editorTitle.textAlignment = UITextAlignmentCenter;
#endif

      editorTitle.textColor = [UIColor blackColor];
      editorTitle.backgroundColor = [UIColor clearColor];
      [self addSubview : editorTitle];

      const CGRect scrollFrame = CGRectMake(10.f, 45.f, [EditorView scrollWidth], frame.size.height - 55.f);
      scrollView = [[ScrollViewWithPickers alloc] initWithFrame : scrollFrame];
      scrollView.backgroundColor = [UIColor clearColor];
      scrollView.autoresizingMask = UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleBottomMargin;
      scrollView.bounces = NO;
      [self addSubview : scrollView];
      self.opaque = NO;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect)rect
{
   //Draw main editor's view as a semi-transparent
   //gray view with rounded corners.

   CGContextRef ctx = UIGraphicsGetCurrentContext();
   if (!ctx) {
      NSLog(@"[EditorView drawRect:], ctx is nil");
      return;
   }

   UIColor *background = [[UIColor scrollViewTexturedBackgroundColor] colorWithAlphaComponent : [EditorView editorAlpha]];
//   UIColor *background = [[UIColor colorWithPatternImage:[UIImage imageNamed:@"inspector_bkn.png"]] colorWithAlphaComponent : [EditorView editorAlpha]];
   CGContextSetFillColorWithColor(ctx, background.CGColor);
   CGContextSetPatternPhase(ctx, CGSizeMake(-8.f, 0.f));

   //Draw the rect with rounded corners now.
   CGContextFillRect(ctx, CGRectMake(0.f, [EditorView ncHeight] / 2, [EditorView ncWidth] / 2, rect.size.height - [EditorView ncHeight]));
   CGContextFillRect(ctx, CGRectMake([EditorView ncWidth] / 2, 0.f, rect.size.width - [EditorView ncWidth] / 2, rect.size.height));

   //Draw arcs.
   CGContextBeginPath(ctx);
   CGContextMoveToPoint(ctx, [EditorView ncWidth] / 2, [EditorView ncHeight] / 2);
   CGContextAddArc(ctx, [EditorView ncWidth] / 2, [EditorView ncHeight] / 2, [EditorView ncWidth] / 2, M_PI, 3 * M_PI / 2, 0);
   CGContextFillPath(ctx);
   //
   CGContextBeginPath(ctx);
   CGContextMoveToPoint(ctx, [EditorView ncWidth] / 2, rect.size.height - [EditorView ncHeight] / 2);
   CGContextAddArc(ctx, [EditorView ncWidth] / 2, rect.size.height - [EditorView ncHeight] / 2, [EditorView ncWidth] / 2, M_PI / 2, M_PI, 0);
   CGContextFillPath(ctx);
}

//____________________________________________________________________________________________________
- (void) removeAllEditors
{
   //Remove all sub-editors.
   for (unsigned i = 0; i < nEditors; ++i) {
      [plates[i] removeFromSuperview];
      [containers[i] removeFromSuperview];
   }

   nEditors = 0;
   currentState = 0;
   nStates = 0;
   animation = NO;
}

//____________________________________________________________________________________________________
- (void) propertyUpdated
{
}

//____________________________________________________________________________________________________
- (CGFloat) recalculateEditorGeometry
{
   const CGFloat dY = 3.f;//space between controls.
   for (unsigned i = 0; i < nStates; ++i) {
      CGFloat currentY = 0.f;
      for (unsigned j = 0; j < nEditors; ++j) {
         plateYs[nEditors * i + j] = currentY;
         currentY += plates[j].frame.size.height + dY;

         const unsigned editorBit = 1 << j;
         if (i & editorBit) {//In this state, j-th editor is visible.
            viewYs[nEditors * i + j] = currentY;
            currentY += views[j].frame.size.height + dY;
         } else
            viewYs[nEditors * i + j] = 0.f;//coordinate is not used, currentY does not need update.
      }
   }

   //Now, the total container height.
   CGFloat totalHeight = 0.f;
   for (unsigned i = 0; i < nEditors; ++i)
      totalHeight += plates[i].frame.size.height + dY + views[i].frame.size.height + dY;

   return totalHeight;
}

//____________________________________________________________________________________________________
- (void) correctFrames
{
   CGRect frame = self.frame;
   frame.origin.x = 10.f;
   frame.origin.y = 45.f;
   frame.size.height -= 55.f;
   frame.size.width = 250.f;
   scrollView.frame = frame;

   const CGFloat totalHeight = [self recalculateEditorGeometry];
   scrollView.contentSize = CGSizeMake([EditorView scrollWidth], totalHeight);
   scrollView.contentOffset = CGPointZero;
}

//____________________________________________________________________________________________________
- (void) setPlatesYs
{
   //Plates positions for the current state.
   for (unsigned i = 0; i < nEditors; ++i) {
      CGRect frame = plates[i].frame;
      frame.origin.y = plateYs[nEditors * currentState + i];
      plates[i].frame = frame;
   }
}

//____________________________________________________________________________________________________
- (void) addSubEditor:(UIView *)element withName : (NSString *)name
{
   if (nEditors == evMaxComponents) {
      NSLog(@"Could not add more editors");
      return;
   }

   //Add this new editor.
   views[nEditors] = element;
   //1. Plate with editor's name and triangle, showing the editor's state.
   //topView is 'self' - the view, which will be informed, that user tapped on editor's plate.
   plates[nEditors] = [[EditorPlateView alloc] initWithFrame : CGRectMake(0.f, 0.f, [EditorView scrollWidth], [EditorPlateView plateHeight]) editorName : name topView : self];
   [scrollView addSubview : plates[nEditors]];

   //Create a container view for sub-editor.
   CGRect elementFrame = element.frame;
   elementFrame.origin = CGPointZero;
   containers[nEditors] = [[UIView alloc] initWithFrame : elementFrame];
   element.frame = elementFrame;
   //Place sub-editor into the container view.
   [containers[nEditors] addSubview : element];
   //Clip to bounds: when we animate sub-editor (appear/disappera)
   //it moves from/to negative coordinates and this negative part
   //should not be visible.
   containers[nEditors].clipsToBounds = YES;
   //Initially, container with sub-editor is hidden.
   containers[nEditors].hidden = YES;
   //Add container.
   [scrollView addSubview : containers[nEditors]];

   //New number of sub-editors and possible editor states.
   ++nEditors;
   nStates = 1 << nEditors;

   //Recalculate possible positions of all plates and containers.
   const CGFloat totalHeight = [self recalculateEditorGeometry];
   //Set scrollView.contentSize to include all sub-editors in opened state.
   scrollView.contentSize = CGSizeMake([EditorView scrollWidth], totalHeight);
   scrollView.contentOffset = CGPointZero;

   //No sub-editor is visible.
   currentState = 0;
   //Also, make new sub-editor transparent.
   element.alpha = 0.f;

   //Place all plates.
   [self setPlatesYs];
}

//____________________________________________________________________________________________________
- (void) presetViewsYs
{
   //These are sub-views positions before animation.
   for (unsigned i = 0; i < nEditors; ++i) {
      //If view must appear now:
      if (containers[i].hidden && (currentState & (1 << i))) {
         CGRect frame = views[i].frame;
         frame.origin.y = viewYs[currentState * nEditors + i];
         //Place the container in a correct position.
         containers[i].frame = frame;
         //Contained view is shifted in a container (it will appear from nowhere).
         frame.origin.y = -frame.size.height;
         views[i].frame = frame;
      }
   }
}

//____________________________________________________________________________________________________
- (void) setViewsYs
{
   //These are new sub-views positions at the end of animation.
   for (unsigned i = 0; i < nEditors; ++i) {
      CGRect frame = views[i].frame;
      if (currentState & (1 << i)) {//View will become visible now (and could be visible before).
         frame.origin.y = viewYs[currentState * nEditors + i];
         containers[i].frame = frame;
         frame.origin.y = 0.;
         views[i].frame = frame;
      } else if (!views[i].hidden) {//View will hide now - it moves outside of container.
         frame.origin.y = -frame.size.height;
         views[i].frame = frame;
      }
   }
}

//____________________________________________________________________________________________________
- (void) setViewsAlphaAndVisibility
{
   //During animation, if view will appear it's alpha changes from 0 to 1,
   //and if it's going to disappear - from 1 to 0.
   //Also, I have to animate small triangle, which
   //shows editor's state (hidden/visible).
   for (unsigned i = 0; i < nEditors; ++i) {
      EditorPlateView *p = (EditorPlateView *)plates[i];
      UIView *v = views[i];
      const BOOL nowVisible = currentState & (1 << i);
      if (containers[i].hidden) {
         if (nowVisible) {
            containers[i].hidden = NO;
            v.alpha = 1.f;
            p.arrowImageView.transform = CGAffineTransformMakeRotation(M_PI / 2);//rotate the triangle.
         }
      } else {
         if (!nowVisible) {
            p.arrowImageView.transform = CGAffineTransformMakeRotation(0.f);//rotate the triangle.
            v.alpha = 0.f;
         }
      }
   }
}

//____________________________________________________________________________________________________
- (void) hideViews
{
   for (unsigned i = 0; i < nEditors; ++i) {
      if (!(currentState & (1 << i)))
         containers[i].hidden = YES;
   }
}

//____________________________________________________________________________________________________
- (void) showEditorFrame
{
   CGRect frameToShow = CGRectMake(0.f, 0.f, 250.f, 90.f);

   if (newOpened != -1) {
      frameToShow = containers[newOpened].frame;
      frameToShow.origin.y = viewYs[currentState * nEditors + newOpened] - 70.f;
      frameToShow.size.height += 70.f;
   }

   [scrollView scrollRectToVisible : frameToShow animated : YES];
   animation = NO;
}

//____________________________________________________________________________________________________
- (void) animateEditor
{
   animation = YES;

   [self presetViewsYs];

   [UIView beginAnimations : nil context : nil];
   [UIView setAnimationDuration : 0.25];
   [UIView setAnimationCurve : UIViewAnimationCurveEaseOut];

   [self setPlatesYs];
   [self setViewsYs];
   [self setViewsAlphaAndVisibility];

   [UIView commitAnimations];

   //Do not hide the views immediately, so user can see animation.
   [NSTimer scheduledTimerWithTimeInterval : 0.15 target : self selector : @selector(hideViews) userInfo : nil repeats : NO];
   [NSTimer scheduledTimerWithTimeInterval : 0.3 target : self selector : @selector(showEditorFrame) userInfo : nil repeats : NO];
}

//____________________________________________________________________________________________________
- (void) plateTapped : (EditorPlateView *) plate
{
   if (animation)
      return;
   //User has tapped on editor's plate.
   //Depending on the current editor's state,
   //we open or close it with animation.
   newOpened = -1;

   for (unsigned i = 0; i < nEditors; ++i) {
      if (plate != plates[i]) {
         if (currentState & (1 << i))
            newOpened = i;//Remember the index of opened editor above our plate.
      } else {
         currentState ^= (1 << i);//reset the bit for the editor.

         if (currentState & (1 << i))//plate's editor was opened.
            newOpened = i;
         else if (currentState) {//Is any editor opened at all?
            //plate's editor was closed, find the next which we can be made visible.
            if (newOpened == -1) {//we did not find any opened editor above the plate yet.
               for (unsigned j = i + 1; j < nEditors; ++j) {
                  if (currentState & (1 << j)) {
                     newOpened = j;
                     break;
                  }
               }
            }
         }

         [self animateEditor];
         break;
      }
   }
}

//____________________________________________________________________________________________________
- (void) setEditorTitle : (const char*) title
{
   editorTitle.text = [NSString stringWithFormat : @"%s", title];
}

@end
