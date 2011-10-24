#include "Constants.h"

namespace ROOT_IOSBrowser {

/////////////Geometric constants for ROOTObjectController.
//'L' postfix is for landscape, 'P' is for portrait.
////Main view (self.view):

//X and Y are the same for portrait and landscape orientation.
const float viewX = 0.f;
const float viewY = 0.f;

//portrait - 768 x 1004 (20 Y pixels are taken by iOS).
const float viewWP = 768.f;
const float viewHP = 1004.f;

//landscape - 1024 x 748 (20 Y pixels are taken by iOS).
const float viewWL = 1024.f;
const float viewHL = 748.f;

////Scroll view:

//X and Y are the same for landscape and portrait.
const float scrollX = 0.f;
const float scrollY = 44.f;//Navigation bar height.

//portrait - 768 x 960 (44 Y pixels from parent are taken by navigation bar).
const float scrollWP = 768.f;
const float scrollHP = 960.f;

//landscape - 1024 x 704 (44 Y pixels from parent are taken by navigation bar).
const float scrollWL = 1024.f;
const float scrollHL = 704.f;

//Default pad's width and height,
//when not zoomed, without editor
//or with editor in landscape orientation.
const float padW = 600.f;
const float padH = 600.f;

//This is pad's width and height, when
//pad is not zoomed and editor is visible,
//device orientation is portrait.
const float padWSmall = 600.f;//30.08 : I decided to make the sizes equal for all possible orientations and states.
const float padHSmall = 600.f;

const float padXNoEditorP = scrollWP / 2 - padW / 2;
const float padYNoEditorP = scrollHP / 2 - padH / 2;
const float padXNoEditorL = scrollWL / 2 - padW / 2;
const float padYNoEditorL = scrollHL / 2 - padH / 2;

//X and Y for pad (no zoom) with editor in portrait orientation:
const float padXWithEditorP = 20.f;
const float padYWithEditorP = scrollHP / 2 - padHSmall / 2;

//X and Y for pad (no zoom) with editor in landscape orientation:
const float padXWithEditorL = 100.f;
const float padYWithEditorL = scrollHL / 2 - padH / 2;

const double predefinedFillColors[nROOTDefaultColors][3] = 
{
{1., 1., 1.},
{0., 0., 0.},
{251 / 255., 0., 24 / 255.},
{40 / 255., 253 / 255., 44 / 255.},
{31 / 255., 29 / 255., 251 / 255.},
{253 / 255., 254 / 255., 52 / 255.},
{253 / 255., 29 / 255., 252 / 255.},
{53 / 255., 1., 254 / 255.},
{94 / 255., 211 / 255., 90 / 255.},
{92 / 255., 87 / 255., 214 / 255.},
{135 / 255., 194 / 255., 164 / 255.},
{127 / 255., 154 / 255., 207 / 255.},
{211 / 255., 206 / 255., 138 / 255.},
{220 / 255., 185 / 255., 138 / 255.},
{209 / 255., 89 / 255., 86 / 255.},
{147 / 255., 29 / 255., 251 / 255.}
};

const unsigned colorIndices[nROOTDefaultColors] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 38, 41, 42, 50, 51};

const unsigned stippleBase = 3000;

}
