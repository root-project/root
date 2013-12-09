#ifndef CONSTANTS_INCLUDED
#define CONSTANTS_INCLUDED

namespace ROOT {
namespace iOS {
namespace Browser {

///////////////////////////////////////////////////////////////////////
//Pad's geometry: sizes, positioins.
//'L' postfix is for landscape, 'P' is for portrait
//device orientation.

////Main view (self.view for ROOTObjectController):

//X and Y are the same for portrait and landscape orientation.
extern const float viewX;
extern const float viewY;

//portrait - 768 x 1004 (20 Y pixels are taken by iOS).
extern const float viewWP;
extern const float viewHP;

//landscape - 1024 x 748 (20 Y pixels are taken by iOS).
extern const float viewWL;
extern const float viewHL;

////Scroll view:

//X and Y are the same for landscape and portrait.
extern const float scrollX;
extern const float scrollY;

//portrait - 768 x 960 (44 Y pixels from parent are taken by navigation bar).
extern const float scrollWP;
extern const float scrollHP;

//landscape - 1024 x 704 (44 Y pixels from parent are taken by navigation bar).
extern const float scrollWL;
extern const float scrollHL;

//Default pad's width and height,
//when not zoomed, without editor
//or with editor in landscape orientation.
extern const float padW;
extern const float padH;

//This is pad's width and height, when
//pad is not zoomed and editor is visible,
//device orientation is portrait.
extern const float padWSmall;
extern const float padHSmall;

extern const float padXNoEditorP;
extern const float padYNoEditorP;
extern const float padXNoEditorL;
extern const float padYNoEditorL;

//X and Y for pad (no zoom) with editor in portrait orientation:
extern const float padXWithEditorP;
extern const float padYWithEditorP;

//X and Y for pad (no zoom) with editor in landscape orientation:
extern const float padXWithEditorL;
extern const float padYWithEditorL;

///////////////////////////////////////////////////////////////////////
//Editor's constants;
enum {

nROOTDefaultColors = 16

};

extern const double predefinedFillColors[nROOTDefaultColors][3];
//Color indices in a standard ROOT's color selection control.
extern const unsigned colorIndices[nROOTDefaultColors];


//Constant for TAttFill, never defined in ROOT, I have to define it here.
extern const unsigned stippleBase;

//This does not have const qualifier, but set only once during startup (by application delegate).
extern bool deviceIsiPad3;

}//namespace Browser
}//namespace iOS
}//namespace ROOT

#endif
