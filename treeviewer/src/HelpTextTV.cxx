// Author : Andrei Gheata	02/10/00

#include "HelpTextTV.h"

#ifndef WIN32
const char gTVHelpAbout[] 		= "\
   TTreeView is GUI version of TTreeViewer, designed to handle ROOT trees and\n\
to take advantage of TTree class features in a graphical manner. It uses only\n\
ROOT native GUI widgets and has capability to work with several trees in the\n\
same session. It provides the following functionalities :\n\n\
  - browsing all root files in the working directory and mapping trees inside;\n\
  - once a tree is mapped, the user can browse branches and work with the\n\
  corresponding sub-branches if there is no need for the whole tree;\n\
  - fast drawing of branches by double-click;\n\
  - easy edit the expressions to be drawn on X, Y and Z axis and/or selection;\n\
  - dragging expressions to one axis and aliasing of expression names;\n\
  - handle input/output event lists;\n\
  - usage of predefined compatible drawing options;\n\
  - possibility of executing user commands and macros and echoing of the current\n\
  command;\n\
  - possibility of interrupting the current command or the event loop (not yet);\n\
  - possibility of selecting the tree entries to be processed (not yet);\n\
  - take advantage of TTree class features via context menu;\n\n\
";


const char gTVHelpStart[]		= "\
   The quickest way to start the tree viewer is to start a ROOT session in \n\
your working directory where you have the root files containing trees.\n\
You will need first to load the library for TTreeView and optionally other\n\
libraries for user defined classes (you can do this later in the session) :\n\
   root [0] gSystem->Load(\"TTreeView\");\n\
   root [1] new TTreeView;\n\
or, to load the tree Mytree from the file Myfile :\n\
   root [1] TFile file(\"Myfile\");\n\
   root [2] new TTreeView(\"Mytree\");\n\n\
This will work if uou have the path to the library TTreeView defined in your\n\
.rootrc file.\n\n
";


const char gTVHelpLayout[]		= "\
   The layout has the following items :\n\n\
  - a menu bar with entries : File, Edit, Run, Options and Help;\n\
  - a toolbar in the upper part where you can issue user commands, change\n\
  the drawing option and the histogram name, two check buttons Hist and Rec\n\
  which toggles histogram drawing mode and command recording respectively;\n\
  - a button bar in the lower part with : buttons DRAW/STOP that issue histogram\n\
  drawing and stop the current command respectively, two text widgets where \n\
  input and output event lists can be specified, a message box and a RESET\n\
  button on the right that clear edited expression content (see Editing...)\n\
  - a tree-type list on the main left panel where you can browse the root files\n\
  from the working directory and load the trees inside by double clicking.\n\
  When the first tree is loaded, a new item called \"TreeList\" will pop-up on\n\
  the list menu and will have the selected tree inside with all branches mapped\n\
  Mapped trees are provided with context menus, activated by right-clicking;\n\
  - a view-type list on the main right panel. The first column contain X, Y and\n\
  Z expression items, an optional cut and ten optional editable expressions.\n\
  The other items in this list are activated when a mapped item from the\n\
  \"TreeList\" is left-clicked (tree or branch) and will describe the conyent\n\
  of the tree (branch). Expressions and leaf-type items can be dragged or\n\
  deleted. A right click on the list-box or item activates a general context\n\
  menu.\n\n\
";


const char gTVHelpBrowse[]		= "\
   Browsing root files from the working directory :\n\n\
Just double-click on the directory item on the left and you will see all\n\
root files from this directory. Do it once more on the files with a leading +\n\
and you will see the trees inside. If you want one or more of those to\n\
be loaded, double-click on them and they will be mapped in a new item called\n\
\"TreeList\".\n\n\
   Browsing trees :\n\n\
Left-clicking on trees from the TreeList will expand their content on the list\n\
from the right side. Double-clicking them will also open their content on the\n\
left, where you can click on branches to expand them on the right.\n\n\ 
";


const char gTVHelpDraggingItems[]	= "\
   Items that can be dragged from the list in the right : expressions and \n\
leaves. Dragging an item and dropping to another will copy the content of first\n\
to the last (leaf->expression, expression->expression). Items far to the right\n\
side of the list can be easily dragged to the left (where expressions are\n\
placed) by dragging them to the left at least 10 pixels.\n\n\ 
";


const char gTVHelpEditExpressions[]	= "\
   All editable expressions from the right panel has two components : a\n\
true name (that will be used when TTree::Draw() commands are issued) and an\n\
alias (used for labeling axes - not yet). The visible name is the alias if\n\
there is one and the true name otherwise.\n\n\
   The expression editor can be activated by right clicking on an\n\
expression item via the command EditExpression from the context menu.\n\
An alternative is to use the Edit-Expression menu after the desired expression\n\
is selected. The editor will pop-up in the left part, but it can be moved.\n\
The editor usage is the following :\n\
  - you can write C expressions made of leaf names by hand or you can insert\n\
  any item from the right panel by clicking on it (recommandable);\n\
  - you should write the item alias by hand since it not ony make the expression\n\
  meaningfull, but it also highly improve the layout for big expressions\n\n\ 
";


const char gTVHelpUserCommands[]	= "\
   User commands can be issued directly from the textbox labeled \"Command\"\n\
from the upper-left toolbar by typing and pressing Enter at the end.\n\
   An other way is from the right panel context menu : ExecuteCommand.\n\n\
All commands can be interrupted at any time by pressing the STOP button\n\
from the bottom-left (not yet)\n\n\
You can toggle recording of the current command in the history file by\n\
checking the Rec button from the top-right\n\n\ 
";


const char gTVHelpContext[]		= "\
   You can activate context menus by right-clicking on items or inside the\n\
box from the right.\n\n\
Context menus for mapped items from the left tree-type list :\n\n\
  The items from the left that are provided with context menus are tree and\n\
branch items. You can directly activate the *MENU* marked methods of TTree\n\
from this menu.\n\n\
Context menu for the right panel :\n\n\
  A general context manu of class TTreeView is acivated if the user\n\
right-clicks the right panel. Commands are :\n\
  - ClearAll        : clears the content of all expressions;\n\
  - ClearExpression : clear the content of the clicked expression;\n\
  - EditExpression  : pops-up the expression editor;\n\
  - ExecuteCommand  : execute a user command;\n\
  - MakeSelector    : equivalent of TTree::MakeSelector();\n\
  - Process         : equivalent of TTree::Process();\n\
  - RemoveExpression: removes clicked item from the list;\n\n\ 
";


const char gTVHelpDrawing[]		= "\
Fast variable drawing : Just double-click an item from the right list.\n\n\
Normal drawing : Edit the X, Y, Z fields as you wish, fill input-output list\n\
names if you have them. You can change output histogram name, or toggle Hist\n\
and Scan modes by checking the corresponding buttons.\n\
  Hist mode implies that the current histogram will be redrawn with the current\n\
graphics options, while Scan mode implies TTree::Scan()\n\n\
  You have a complete list of histogram options in multicheckable lists from\n\
the Option menu. When using this menu, only the options compatible with the\n\
current histogram dimension will be available. You can check multiple options\n\
and reset them by checking the Default entries.\n\n\
  After completing the previous operations you can issue the draw command by\n\
pressing the DRAW button.\n\n\ 
";


const char gTVHelpMacros[]		= "\
   Macros can be loaded and executed in this version only by issuing\n\
the corresponding user commands (see help on user commands)\n\n\
";
#else
const char gTVHelpAbout[] 		= "empty";
const char gTVHelpStart[]		= "empty";
const char gTVHelpLayout[]		= "empty";
const char gTVHelpBrowse[]		= "empty";
const char gTVHelpDraggingItems[]	= "empty";
const char gTVHelpEditExpressions[]	= "empty";
const char gTVHelpUserCommands[]	= "empty";
const char gTVHelpLoopingEvents[]	= "empty";
const char gTVHelpDrawing[]		= "empty";
const char gTVHelpMacros[]		= "empty";

#endif
