// @(#)root/gui:$Name:  $:$Id: HelpText.cxx,v 1.2 2000/09/08 07:41:00 brun Exp $
// Author: Fons Rademakers   28/07/97

#include "HelpText.h"

#if !defined(WIN32) || defined(GDK_WIN32)
const char gHelpAbout[] = "\
ROOT is an OO framework for large scale scientific data\n\
analysis and data mining. It has been developed at CERN with the\n\
sponsorship of HP and is currently being used by a number of large\n\
high energy physics experiments. The ROOT system, written in C++,\n\
contains, among others, an efficient hierarchical OO database, a\n\
C++ interpreter, advanced statistical analysis (multi dimensional\n\
histogramming, fitting and minimization algorithms) and visualization\n\
tools. The user interacts with ROOT via a graphical user interface, the\n\
command line or batch scripts. The command and scripting language\n\
is C++ (using the interpreter) and large scripts can be compiled and\n\
dynamically linked in. Using the PROOF (Parallel ROOT Facility)\n\
extension large databases can be analysed in parallel on MPP's,\n\
SMP's or loosely coupled workstation/PC clusters. ROOT also\n\
contains a C++ to HTML documentation generation system using\n\
the interpreter's dictionaries (the reference manual on\n\
the web is generated that way) and a rich set of interprocess\n\
communication classes allowing the transfer of complete objects\n\
from one process to another.\n\
";


const char gHelpBrowser[] = "\
The ROOT general object browser (see TBrowser) can be used to browse collections\n\
such as the list of classes, geometries, files and TTrees. A browser can be \n\
started from the Start Browser item in the canvas View menu or by creating a \n\
browser object. More than one browser can be active at any time. \n\
A Browser window is divided in two parts:\n\
  - a left window showing the collections that can be browsed.\n\
  - a right window with the elements of a given collection.\n\
Double clicking on the icons in the right window performs a default action\n\
specific to the object. For example, clicking on a histogram icon will\n\
display the histogram. Clicking on a TTree variable will histogram and\n\
display this variable. Clicking on an icon with the right mouse button\n\
displays a context menu like for objects in a canvas.\n\
The following collections can be browsed:\n\
  - The class structures\n\
  - The detector geometries\n\
  - The ROOT files\n\
  - the ROOT mapped files (shared memory)\n\
A user-defined collection (TList,etc) can be added in the left window via:\n\
  gROOT->GetListOfBrowsables()->Add(list,title).\n\n\
";


const char gHelpGLViewer[] = "\
     PRESS \n\
     \tu\t--- to Move down \n\
     \ti\t--- to Move up\n\
     \th\t--- to Shift right\n\
     \tl\t--- to Shift left\n\
     \tj\t--- to Pull the object backward\n\
     \tk\t--- to Push the object foreward\n\n\
     \t+\t--- to Increase speed to move\n\
     \t-\t--- to Decrease speed to move\n\n\
     \tn\t--- to turn \"SMOOTH\" color mode on\n\
     \tm\t--- to turn \"SMOOTH\" color mode off\n\n\
     \tt\t--- to toggle Light model\n\n\
     \tp\t--- to toggle Perspective/Orthographic projection\n\
     \tr\t--- to Hidden surface mode\n\
     \tw\t--- to wireframe mode\n\
     \tc\t--- to cull-face mode\n\n\
     \ts\t--- to increase the scale factor (clip cube borders)\n\
     \ta\t--- to decrease the scale factor (clip cube borders)\n\n\
     HOLD the left mouse button and MOVE mouse to ROTATE object\n\n\
";


const char gHelpPostscript[] = "\
To generate a Postscript (or encapsulated ps) file corresponding to\n\
a single image in a canvas, you can:\n\
 -Select the Print PostScript item in the canvas File menu.\n\
  By default, a Postscript file with the name of the canvas.ps is generated.\n\n\
 -Click in the canvas area, near the edges, with the right mouse button\n\
  and select the Print item. You can select the name of the Postscript\n\
  file. If the file name is xxx.ps, you will generate a Postscript file named\n\
  xxx.ps. If the file name is xxx.eps, you generate an encapsulated Postscript\n\
  file instead.\n\n\
 -In your program (or macro), you can type:\n\
    c1->Print(\"xxx.ps\") or c1->Print(\"xxx.eps\")\n\
  This will generate a file corresponding to the picture in the canvas\n\
  pointed by c1.\n\n\
 -pad1->Print(\"xxx.ps\")\n\
  prints only the picture in the pad pointed by pad1. The size\n\
  of the Postcript picture, by default, is computed to keep the aspect ratio\n\
  of the picture on the screen, where the size along x is always 20cm. You\n\
  can set the size of the PostScript picture before generating the picture\n\
  with a command such as: gStyle->SetPaperSize(xsize,ysize) (size in cm).\n\n\
";


const char gHelpButtons[] = "\
Once objects have been drawn in a canvas, they can be edited/moved\n\
by pointing directly to them. The cursor shape is changed\n\
to suggest the type of action that one can do on this object.\n\
Clicking with the right mouse button on an object pops-up\n\
a contextmenu with a complete list of actions possible on this object.\n\n\
When the mouse is moved or a button pressed/released, the TCanvas::HandleInput\n\
function scans the list of objects in all its pads and for each object\n\
invokes object->DistancetoPrimitive(px, py). This function computes\n\
a distance to an object from the mouse position at the pixel\n\
position px,py and return this distance in pixel units. The selected object\n\
will be the one with the shortest computed distance. To see how this work,\n\
select the \"Event Status\" item in the canvas \"Options\" menu.\n\
ROOT will display one status line showing the picked object. If the picked\n\
object is, for example an histogram, the status line indicates the name\n\
of the histogram, the position x,y in histogram coordinates, the channel\n\
number and the channel content.\n\n\
If you click on the left mouse button, the object->ExecuteEvent(event,px,py)\n\
function is called.\n\n"
"If you click with the right mouse button, a context menu (see TContextMenu)\n\
with the list of possible actions for this object is shown. You will notice\n\
that most graphics objects derive from one or several attribute classes \n\
TAttLine, TAttFill, TAttText or TAttMarker.\n\
You can edit these attributes by selecting the corresponding item in the pop-up\n\
menu. For example selecting SetFillAttributes displays a panel\n\
with the color palette and fill area types. The name and class of the object\n\
being edited is shown in the bar title of the panel.\n\n\
The middle button (or left+right on a 2-buttons mouse) can be used to change\n\
the current pad to the pointed pad. The current pad is always highlighted.\n\
Its frame is drawn with a special color.\n\
A canvas may be automatically divided into pads via TPad::Divide.\n\
When a canvas/pad is divided, one can directly set the current path to one of \n\
the subdivisions by pointing to this pad with the middle button. For example:\n\
  c1.Divide(2,3); // create 6 pads (2 divisions along x, 3 along y).\n\
  To set the current pad to the bottom right pad, do  c1.cd(6);\n\
Note that c1.cd() is equivalent to c1.cd(0) and sets the current pad\n\
to c1 itself.\n\n\
";


const char gHelpGraphicsEditor[] = "\
ROOT has a simple built-in graphics editor to draw and edit basic primitives\n\
starting from an empty canvas or on top of a picture (eg. histogram).\n\
The editor is started by selecting the \"Editor\" item in the\n\
canvas \"Edit\" menu. A menu appears into an independent window.\n\
You can create the following graphics objects:\n\
 -An arc of circle. Click on the center of the arc, then move the mouse.\n\
  A rubberband circle is shown. Click again with the left button to freeze\n\
  the arc.\n\n\
 -A line segment. Click with the left button on the first and last point.\n\n\
 -An arrow. Click with the left button at the point where you want to start\n\
  the arrow, then move the mouse and click again with the left button\n\
  to freeze the arrow.\n\n\
 ->A Diamond. Click with the left button and freeze again with the left button.\n\
   The editor draws a rubber band box to suggest the outline of the diamond.\n\n\
 -An Ellipse. Proceed like for an arc.\n\
  You can grow/shrink the ellipse by pointing to the sensitive points.\n\
  They are highlighted. You can move the ellipse by clicking on the ellipse,\n\
  but not on the sensitive points. If, with the ellipse context menu,\n\
  you have selected a fill area color, you can move a filled-ellipse by\n\
  pointing inside the ellipse and dragging it to its new position.\n\
  Using the contextmenu, you can build an arc of ellipse and tilt the ellipse.\n\n\
 -A Pad. Click with the left button and freeze again with the left button.\n\
  The editor draws a rubber band box to suggest the outline of the pad.\n\n\
 -A PaveLabel. Proceed like for a pad. Type the label to be put in the box. \n\
  Then type carriage return. The text will be redrawn to fill the box.\n\n"
" -A PaveText or PavesText. Proceed like for a pad.\n\
  You can then click on the PaveText object with the right mouse button\n\
  and select the option AddText.\n\n\
 -A PolyLine. Click with the left button for the first point,\n\
  move the mouse, click again with the left button for a new point. Close\n\
  the polyline by clicking twice at the same position.\n\
  To edit one vertex point, pick it with the left button and drag to the \n\
  new point position.\n\n\
 -A Curly/Wavy line. Click with the left button on the first and last point.\n\
  You can use the context menu to set the wavelength or amplitude.\n\n\
 -A Curly/Wavy arc. Click with the left button on the arc center and click again\n\
  to stop at the arc radius.\n\n\
  You can use the context menu to set the wavelength or amplitude.\n\
  You can use the context menu to set the phimin and phimax.\n\n\
 -A Text/Latex string. Click with the left button where you want to draw the text, \n\
  then type the text terminated by carriage return. To move the text, point\n\
  on it keeping the left mouse button pressed and drag the text to its new \n\
  position. You can grow/shrink the text if you position the mouse to the first\n\
  top-third part of the string, then move the mouse up or down to grow or \n\
  shrink the text respectively. If you position near the bottom-end of the text,\n\
  you can rotate it.\n\n\
 - A Marker. Click with the left button where to place the marker.\n\
  The marker by default can be modified by gStyle->SetMarkerStyle().\n\n\
 -A Graphical Cut. Click with the left button on each point of a polygone\n\
  delimiting the selected area. Close the cut by clicking twice at the\n\
  same position. A TCutG object is created. It can be used\n\
  as a selection for TTree::Draw. You can get a pointer to this object with\n\
  TCutG *cut = (TCutG*)gPad->FindObject(\"CUTG\").\n\n\
  ";


const char gHelpPullDownMenus[] = "\
Each canvas has a toolbar menu with the following items:\n\
  \"File\" with the items:\n\
     <Save as Canvas.C>    generates a C++ macro to reproduce the canvas\n\
     <save as Canvas.ps>   makes a Postscript file\n\
     <save as Canvas.eps>  makes a Postscript encapsulated file\n\
     <save as Canvas.gif>  makes a GIF file\n\
     <save as Canvas.root> saves canvas objects in a Root file\n\n\
  \"Edit\" with the items:\n\
     <Editor>  invokes the graphics editor (see below).\n\
     <Clear Pad>  clears the last selected pad (using middle mouse button)\n\
     <Clear canvas> clears this canvas.\n\n\
  \"View\" with the items:\n\
     <Colors>  creates a new canvas showing the color palette.\n\
     <Markers> creates a new canvas showing the various marker styles.\n\
     <View with X3D>  If the last selected pad contains a 3-d structure,\n\
                     a new canvas is created. To get help menu, type M.\n\
                     The 3-d picture can be interactively rotated, zoomed\n\
                     in wireframe, solid, hidden line or stereo mode.\n\
     <View with OpenGL>  If the last selected pad contains a 3-d structure,\n\
                     a new canvas is created. See OpenGL canvas help.\n\
                     The 3-d picture can be interactively rotated, zoomed\n\
                     in wireframe, solid, hidden line or stereo mode.\n\n\
   \"Options\" with the items:\n"
"      <Event Status> toggles the identification of the objects when\n\
                     moving the mouse.\n\
      <Statistics>   toggles the display of the histogram statistics box.\n\
      <Histo Title>  toggles the display of the histogram title.\n\
      <Fit Params>   toggles the display of the histogram/graph fit parameters.\n\
      <Can Edit Histograms>   enables/disables the possibility to edit\n\
                             histogram bin contents.\n\
   \"Inspector\" with the items:\n\
      <ROOT>   Inspects the top level gROOT object (in a new canvas).\n\
      <Start Browser> Starts a new object browser (see below).\n\n\
In addition to the tool bar menus, one can set the canvas properties\n\
by clicking with the right mouse button in the regions closed to the canvas \n\
borders. This will display a menu to perform operations on a canvas.\n\n\
";


const char gHelpCanvas[] = "\
A canvas (see TCanvas) is a top level pad (See TPad).\n\
A pad is a linked list of primitives of any type (graphics objects,\n\
histograms, detectors, tracks, etc.). A Pad supports linear and log scales \n\
coordinate systems. It may contain other pads (unlimited pad hierarchy).\n\
Adding a new element into a pad is in general performed by the Draw\n\
member function of the object classes.\n\
It is important to realize that the pad is a linked list of references\n\
to the original object. The effective drawing is performed when the canvas\n\
receives a signal to be painted. This signal is generally sent when typing \n\
carriage return in the command input or when a graphical operation has been \n\
performed on one of the pads of this canvas. When a Canvas/Pad is repainted,\n\
the member function Paint for all objects in the Pad linked list is invoked.\n\
For example, in case of an histogram, the histogram.Draw() operation\n\
only stores a reference to the histogram object and not a graphical\n\
representation of this histogram. When the mouse is used to change (say the bin\n\
content), the bin content of the original histogram is changed !!\n\n\
     Generation of a C++ macro reproducing the canvas\n\
     ************************************************\n\
Once you are happy with your picture, you can select the <Save as canvas.C>\n\
item in the canvas File menu. This will automatically generate a macro with \n\
the C++ statements corresponding to the picture. This facility also works \n\
if you have other objects not drawn with the graphics editor.\n\n\
     Saving the canvas and all its objects in a Root file\n\
     ****************************************************\n\
Select <Save as canvas.root> to save a canvas in a Root file\n\
In another session, one can access the canvas and its objects, eg:\n\
   TFile f(\"canvas.root\")\n\
   canvas.Draw()\n\n\
";


const char gHelpObjects[] = "\
All objects context menus contain the following items:\n\
 -DrawClass. Draw the inheritance tree for a given object. \n\
  A new canvas is created showing the list of classes composing this object.\n\
  For each class, the list of data members and member functions is displayed.\n\n\
 -Inspect. Display the contents of a given object. A new canvas is created\n\
 with a table showing for each data member, its name, current value and its \n\
 comment field. If a data member is a pointer to another object, one can click\n\
 on the pointer and, in turn, inspect the pointed object,etc.\n\n\
 -Dump. Same as Inspect, except that the output is on stdout.\n\n\
";
#else
const char gHelpAbout[]         = "empty";
const char gHelpBrowser[]       = "empry";
const char gHelpGLViewer[]      = "empty";
const char gHelpPostscript[]    = "empty";
const char gHelpButtons[]       = "empty";
const char gHelpGraphicsEditor[]= "empty";
const char gHelpPullDownMenus[] = "empty";
const char gHelpCanvas[]        = "empty";
const char gHelpObjects[]       = "empty";
#endif
