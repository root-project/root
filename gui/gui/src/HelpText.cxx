// @(#)root/gui:$Id$
// Author: Fons Rademakers   28/07/97

#include "HelpText.h"

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
The ROOT general object browser (see TBrowser) can be used to \n\
browse collections such as the list of classes, geometries, files \n\
and TTrees. A browser can be started from the Start Browser item in \n\
the canvas View menu or by creating a browser object. \n\
More than one browser can be active at any time. \n\
A Browser window has three main tabs, separated by horizontal and \n\
vertical splitters.\n\
By default, the left pane contains the file browser, which is the core \n\
of the browser. \n\
From there, user can: \n\
 - Execute Root macros: \n\
   To execute the macro, double-click on the file icon. \n\
   NB: the editor must not be active on the right tab, otherwise the \n\
   macro will be opened in the editor. \n\
 - Open text files in the editor: \n\
   To open the file, double-click on the file icon while the editor \n\
   tab being active on the right tab. \n\
   It is also possible to drag the file from the list tree and drop it \n\
   in the editor. \n\
   Once the file is opened in the editor, if it is a Root Macro, it \n\
   can be executed with the button 'execute' in the editors's tool bar. \n\
 - Display picture files in the canvas: \n\
   Drag the picture file from the list tree and drop it in the canvas. \n\
 - Browse ROOT files: \n\
   To open the file, double-click on the file icon. Its content will be \n\
   displayed in the list tree. \n\
   From there, user can double-click on any item (i.e. histogram) to \n\
   display it in the canvas. \n\
   It is also possible to drag the item from the list tree and drop it \n\
   in the canvas. \n\
 - Browse ROOT files from Web: \n\
   From the 'Browser' menu, select 'New HTML'. A new tab is created, \n\
   containing a HTML browser. \n\
   From there, type the URL from where you want to access Root files. \n\
   Click once on the file you want to open. The file is opened and the \n\
   browser automatically switch to the 'ROOT Files' folder in the list \n\
   tree. Now, just browse the file as a local Root file. \n\
";

const char gHelpBrowserLite[] = "\
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
     \tx X\t--- to Rotate about x\n\
     \ty Y\t--- to Rotate about y\n\
     \tz Z\t--- to Rotate about z\n\n\
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
     HOLD the left mouse button and MOVE mouse to ROTATE object\n\
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
The pad editor can be toggled by selecting the \"Editor\" item in the\n\
canvas \"View\" menu. It appears on the left side of the canvas window.\n\
You can edit the attributes of the selected object via the provided GUI widgets\n\
in the editor frame. The selected object name is displayed in the pad editor\n\
with a set of options available for interactive changing:\n\
 - fill attributes: style and foreground color\n\
 - line attributes: style, width and color\n\
 - text attributes: font, size, align and color\n\
 - marker attributesr: color, style and size\n\
 - a set of axis attributes\n\n\
The buttons for primitive drawing are placed in the tool bar that can be\n\
toggled by selecting the \"Toolbar\" item in the canvas \"View\" menu.\n\
All picture buttons provide tool tips for helping you. Using them\n\
you can create as before the following graphics objects:\n\
 -An arc of circle. Click on the center of the arc, then move the mouse.\n\
  A rubberband circle is shown. Click again with the left button to freeze\n\
  the arc.\n\n\
 -A line segment. Click with the left button on the first and last point.\n\n\
 -An arrow. Click with the left button at the point where you want to start\n\
  the arrow, then move the mouse and click again with the left button\n\
  to freeze the arrow.\n\n\
 -A Diamond. Click with the left button and freeze again with the left button.\n\
   The editor draws a rubber band box to suggest the outline of the diamond.\n\n\
 -An Ellipse. Proceed like for an arc.\n\
  You can grow/shrink the ellipse by pointing to the sensitive points.\n\
  They are highlighted. You can move the ellipse by clicking on the ellipse,\n\
  but not on the sensitive points. If, with the ellipse context menu,\n\
  you have selected a fill area color, you can move a filled-ellipse by\n\
  pointing inside the ellipse and dragging it to its new position.\n\
  Using the contextmenu, you can build an arc of ellipse and tilt the ellipse.\n\n\
 -A Pad. Click with the left button and freeze again with the left button.\n\
  The editor draws a rubber band box to suggest the outline of the pad.\n\n"
" -A PaveLabel. Proceed like for a pad. Type the label to be put in the box. \n\
  Then type carriage return. The text will be redrawn to fill the box.\n\n\
 -A PaveText or PavesText. Proceed like for a pad.\n\
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
  then type the text terminated by carriage return or by escape. To move the text, \n\
  point on it keeping the left mouse button pressed and drag the text to its new \n\
  position. You can grow/shrink the text if you position the mouse to the first\n\
  top-third part of the string, then move the mouse up or down to grow or \n\
  shrink the text respectively. If you position near the bottom-end of the text,\n\
  you can rotate it.\n\n\
 -A Marker. Click with the left button where to place the marker.\n\
  The marker by default can be modified by gStyle->SetMarkerStyle().\n\n\
 -A Graphical Cut. Click with the left button on each point of a polygone\n\
  delimiting the selected area. Close the cut by clicking twice at the\n\
  same position. A TCutG object is created. It can be used\n\
  as a selection for TTree::Draw. You can get a pointer to this object with\n\
  TCutG *cut = (TCutG*)gPad->FindObject(\"CUTG\").\n\n\
  ";


const char gHelpPullDownMenus[] = "\
Each canvas has a menu bar with the following items:\n\
\"File\" with the items:\n\
     <New Canvas  >   opens a new canvas window\n\
     <Open...     >   brings up the Open dialog\n\
     <Close Canvas>   closes the canvas window\n\
     <Save        >   pops up a cascade menu so that you can save the canvas \n\
                      under its current name in the following formats:\n\
        <name.ps  >   makes a Postscript file\n\
        <name.eps >   makes a Postscript encapsulated file\n\
        <name.pdf >   makes a PDF file\n\
        <name.svg >   makes a SVG file\n\
        <name.tex >   makes a TeX file\n\
        <name.gif >   makes a GIF file\n\
        <name.C   >   generates a C++ macro to reproduce the canvas\n\
        <name.root>   saves canvas objects in a Root file\n\
     <Save As...  >   brings up the Save As... dialog\n\
     <Print       >   prints the canvas as a Postscript file canvas_name.ps\n\
     <Quit ROOT   >   stops running the ROOT\n\n\
\"Edit\" with the items:\n\
     <Cut  >          not implemented\n\
     <Copy >          not implemented\n\
     <Paste>          not implemented\n\
     <Clear>          pops up a cascaded menu with the items:\n\
           <Pad   >   clears the last selected pad via middle mouse button)\n\
           <Canvas>   clears this canvas.\n\
     <Undo >          not implemented\n\
     <Redo >          not implemented\n\n"
"\"View\" with the items:\n\
     <Editor      >   toggles the pad editor\n\
     <Toolbar     >   toggles the tool bar\n\
     <Event Status>   toggles the event status bar that shows the identification\n\
                      of the objects when moving the mouse\n\
     <Colors      >   creates a new canvas showing the color palette\n\
     <Fonts       >   not implemented\n\
     <Markers     >   creates a new canvas showing the various marker styles\n\
     <View With   >   pops up a cascaded menu with the items:\n\
           <X3D   >   If the last selected pad contains a 3-d structure,\n\
                      a new canvas is created. To get help menu, type M.\n\
                      The 3-d picture can be interactively rotated, zoomed\n\
                      in wireframe, solid, hidden line or stereo mode.\n\
           <OpenGL>   If the last selected pad contains a 3-d structure,\n\
                      a new canvas is created. See OpenGL canvas help.\n\
                      The 3-d picture can be interactively rotated, zoomed\n\
                      in wireframe, solid, hidden line or stereo mode.\n\n\
\"Options\" with the items:\n\
      <Event Status>  toggles the identification of the objects when\n\
                      moving the mouse.\n\
      <Statistics>    toggles the display of the histogram statistics box.\n\
      <Histo Title>   toggles the display of the histogram title.\n\
      <Fit Params>    toggles the display of the histogram/graph fit parameters.\n\
      <Can Edit Histograms>   enables/disables the possibility to edit\n\
                              histogram bin contents.\n\
\"Inspector\" with the items:\n\
      <ROOT         >  Inspects the top level gROOT object (in a new canvas).\n\
      <Start Browser>  Starts a new object browser (see below).\n\n\
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

const char gHelpTextEditor[] = "\n\
 ____________________________________________________________________\n\
|                                                                    |\n\
|                          TGTextEditor                              |\n\
|____________________________________________________________________|\n\n\
                           Introduction\n\n\
TGTextEditor is a simple text editor that uses the TGTextEdit widget.\n\
It provides all functionalities of TGTextEdit as copy, paste, cut,\n\
search, go to a given line number. In addition, it provides the\n\
possibilities for compiling, executing or interrupting a running\n\
macro.\n\n\
                          Basic Features\n\n\
      New Document\n\n\
To create a new blank document, select File menu / New, or click the\n\
New toolbar button. It will create a new instance of TGTextEditor.\n\n\
      Open/Save File\n\n\
To open a file, select File menu / Open or click on the Open toolbar\n\
button. This will bring up the standard File Dialog for opening files.\n\
If the current document has not been saved yet, you will be asked either\n\
to save or abandon the changes.\n\
To save the file using the same name, select File menu / Save or the\n\
toolbar Save button. To change the file name use File menu / Save As...\n\
or corresponding SaveAs button on the toolbar.\n\n\
      Text Selection\n\n\
You can move the cursor by simply clicking on the desired location\n\
with the left mouse button. To highlight some text, press the mouse\n\
and drag the mouse while holding the left button pressed.\n\
To select a word, double-click on it;\n\
to select the text line - triple-click on it;\n\
to select all - do quadruple-click.\n\n\
      Cut, Copy, Paste\n\n\
After selecting some text, you can cut or copy it to the clipboard.\n\
A subsequent paste operation will insert the contents of the clipboard\n\
at the current cursor location.\n\n"
"      Text Search\n\n\
The editor uses a standard Search dialog. You can specify a forward or\n\
backward search direction starting from the current cursor location\n\
according to the selection made of a case sensitive mode or not.\n\
The last search can be repeated by pressing F3.\n\n\
      Text Font\n\n\
You can change the text font by selecting Edit menu / Set Font.\n\
The Font Dialog pops up and shows the Name, Style and Size of any\n\
available font. The selected font sample is shown in the preview area.\n\n\
      Executing Macros\n\n\
You can execute the currently loaded macro in the editor by selecting\n\
Tools menu / Execute Macro; by clicking on the corresponding toolbar\n\
button, or by using Ctrl+F5 accelerator keys.\n\
This is identical to the command \".x macro.C\" in the root prompt\n\
command line.\n\n\
      Compiling Macros\n\n\
The currently loaded macro can be compiled with ACLiC if you select\n\
Tools menu / Compile Macro; by clicking on the corresponding toolbar\n\
button, or by using Ctrl+F7 accelerator keys.\n\
This is identical to the command \".L macro.C++\" in the root prompt\n\
command line.\n\n\
      Interrupting a Running Macro\n\n\
You can interrupt a running macro by selecting the Tools menu / \n\
Interrupt; by clicking on the corresponding toolbar button, or by \n\
using Shift+F5 accelerator keys.\n\n\
      Interface to CINT Interpreter\n\n\
Any command entered in the 'Command' combo box will be passed to the\n\
CINT interpreter. This combo box will keep the commands history and \n\
will allow you to re-execute the same commands during an editor session.\n\n"
"      Keyboard Bindings\n\n\
The following table lists the keyboard shortcuts and accelerator keys.\n\n\
Key:              Action:\n\
====              =======\n\n\
Up                Move cursor up.\n\
Shift+Up          Move cursor up and extend selection.\n\
Down              Move cursor down.\n\
Shift+Down        Move cursor down and extend selection.\n\
Left              Move cursor left.\n\
Shift+Left        Move cursor left and extend selection.\n\
Right             Move cursor right.\n\
Shift+Right       Move cursor right and extend selection.\n\
Home              Move cursor to begin of line.\n\
Shift+Home        Move cursor to begin of line and extend selection.\n\
Ctrl+Home         Move cursor to top of page.\n\
End               Move cursor to end of line.\n\
Shift+End         Move cursor to end of line and extend selection.\n\
Ctrl+End          Move cursor to end of page.\n\
PgUp              Move cursor up one page.\n\
Shift+PgUp        Move cursor up one page and extend selection.\n\
PgDn              Move cursor down one page.\n\
Shift+PgDn        Move cursor down one page and extend selection.\n\
Delete            Delete character after cursor, or text selection.\n\
BackSpace         Delete character before cursor, or text selection.\n\
Ctrl+B            Move cursor left.\n\
Ctrl+D            Delete character after cursor, or text selection.\n\
Ctrl+E            Move cursor to end of line.\n\
Ctrl+H            Delete character before cursor, or text selection.\n\
Ctrl+K            Delete characters from current position to the end of\n\
                  line.\n\
Ctrl+U            Delete current line.\n\
";

const char gHelpRemote[] = "\
Remote session help:\n\
.R [user@]host[:dir] [-l user] [-d dbg] [[<]script] | [host] -close\n\
Create a ROOT session on the specified remote host.\n\
The variable \"dir\" is the remote directory to be used as working dir.\n\
The username can be specified in two ways, \"-l\" having the priority\n\
(as in ssh). A \"dbg\" value > 0 gives increasing verbosity.\n\
The last argument \"script\" allows to specify an alternative script to\n\
be executed remotely to startup the session, \"roots\" being\n\
the default. If the script is preceded by a \"<\" the script will be\n\
sourced, after which \"roots\" is executed. The sourced script can be \n\
used to change the PATH and other variables, allowing an alternative\n\
\"roots\" script to be found.\n\
To close down a session do \".R host -close\".\n\
To switch between sessions do \".R host\", to switch to the local\n\
session do \".R\".\n\
To list all open sessions do \"gApplication->GetApplications()->Print()\".\n\
";
