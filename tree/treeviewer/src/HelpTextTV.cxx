// @(#)root/treeviewer:$Id$
// Author: Andrei Gheata   02/10/00

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "HelpTextTV.h"

const char gTVHelpAbout[] = "\
The TreeViewer is a graphic user interface designed to handle ROOT\n\
trees and to take advantage of TTree class features. It uses ROOT native\n\
GUI widgets adapted for drag-and-drop functionality. The following\n\
capabilities are making the viewer a helpful tool for analysis:\n\n\
  - several trees may be opened in the same session\n\
  - branches and leaves can be easily browsed or scanned\n\
  - fast drawing of branch expressions by double-clicking\n\
  - new variables/selections easy to compose with the built-in editor\n\
  - histograms can be composed by dragging leaves or user-defined expressions\n\
    to X, Y and Z axis items\n\
  - the tree entries to be processed can be selected with a double slider\n\
  - selections can be defined and activated by dragging them to the <Cut> item\n\
  - all expressions can be aliased and aliases can be used in composing others\n\
  - input/output event lists easy to handle\n\
  - menu with histogram drawing options\n\
  - user commands may be executed within the viewer and the current command\n\
    can be echoed\n\
  - current <Draw> event loop is reflected by a progress bar and may be\n\
    interrupted by the user\n\
  - all widgets have self-explaining tool tips and/or context menus\n\
  - expressions/leaves can be dragged to a <Scan box> and scanned by\n\
    double-clicking this item. The result can be redirected to an ASCII file\n\n\
";

const char gTVHelpStart[] = "\
   1) From the TBrowser:\n\n\
      Select a tree in the TBrowser, then call the StartViewer() method\n\
      from its context menu (right-click on the tree).\n\n\
   2) From the command line:\n\n\
      Start a ROOT session in the directory where you have your tree.\n\
      You will need first to load the library for TTreeViewer and optionally\n\
      other libraries for user defined classes (you can do this later in the\n\
      session):\n\
         root [0] gSystem->Load(\"TTreeViewer\");\n\
      Supposing you have the tree MyTree in the file MyFile.root, you can do:\n\
         root [1] TFile file(\"Myfile.root\");\n\
         root [2] new TTreeViewer(\"Mytree\");\n\
      or:\n\
         root [2] TreeViewer *tv = new TTreeViewer();\n\
         root [3] tv->SetTreeName(\"Mytree\");\n\n\
   NOTE: Once a TTreeViewer is started, one can access it from the interpreter\n\
         using the global identifier gTV\n\n\
";

const char gTVHelpLayout[] = "\
The layout has the following items:\n\n\
  - a menu bar with entries: File, Edit, Run, Options and Help\n\
  - a toolbar in the upper part where you can issue user commands, change\n\
    the drawing option and the histogram name, three check buttons Hist, Rec\n\
    and Scan.HIST toggles histogram drawing mode, REC enables recording of the\n\
    last command issued and SCAN enables redirecting of TTree::Scan command in\n\
    an ASCII file (see Scanning expressions...)\n\
  - a button bar in the lower part with buttons DRAW/STOP that issue\n\
    histogram drawing and stop the current command respectively, two text\n\
    widgets where input and output event lists can be specified, a message\n\
    box and a RESET button on the right that clear edited expression content\n\
    (see Editing...)\n\
  - a tree-type list on the main left panel where you can select among trees\n\
    or branches. The tree/branch will be detailed in the right panel.\n\
    Mapped trees are provided with context menus, activated by right-clicking.\n\
  - a view-type list on the right panel. The first column contain X, Y and\n\
    Z expression items, an optional cut and ten optional editable expressions.\n\
    Expressions and leaf-type items can be dragged or deleted. A right click\n\
    on the list-box or item activates context menus.\n\n\
";

const char gTVHelpOpenSave[] = "\
There are 3 methods of loading a tree in the TreeViewer:\n\n\
   1. If the tree is opened in a TBrowser, one can direcly call TTree::StartViewer()\n\
      from its context menu (righ-button click on the tree).\n\
   If the tree is in a file, one has first to load it into memory:\n\
   - using first <File/Open tree file> menu to open the .root file or just:\n\
      TFile f(\"myFile\");\n\
   - once the file is opened, one can load a tree in the tree viewer:\n\
   2. Knowing the name of the tree:\n\
      gTV->SetTreeName(\"myTree\"); (this method can be also called from the context\n\
                                   menu of the panel on the right)\n\
   3. Getting a pointer to the tree:\n\
      TTree *tree = (TTree*)f.Get(\"myTree\");\n\
      gTV->AppendTree(tree);\n\
NOTE that method 2. calls gROOT->FindObject(\"myTree\") that will retreive the first\n\
tree found with this name, not necesarry from the opened file.\n\n\
Several trees can be opened in the same TreeViewer session :\n\n\
   TFile *f1(\"first.root\",\"READ\");\n\
   TTree *tree1 = (TTree*)f1->Get(\"myTree1\");\n\
   gTV->AppendTree(tree1);\n\
   TFile *f2(\"second.root\",\"READ\");\n\
   TTree *tree2 = (TTree*)f2->Get(\"myTree2\");\n\
   gTV->AppendTree(tree2);\n\n\
To save the current session, use File/SaveSource menu or the SaveSource()\n\
method from the context menu of the right panel (this allows changing the name of the\n\
file)\n\n\
To open a previously saved session for the tree MyTree, first open MyTree\n\
in the browser, then use <File/Open session> menu.\n\n\
";

const char gTVHelpDraggingItems[] = "\
Items that can be dragged from the list in the right: expressions and\n\
leaves. Dragging an item and dropping to another will copy the content of\n\
first to the last (leaf->expression, expression->expression). Items far to\n\
the right side of the list can be easily dragged to the left (where\n\
expressions are placed) by dragging them to the left at least 10 pixels.\n\n\
";

const char gTVHelpEditExpressions[] = "\
Any editable expression from the right panel has two components: a\n\
true name (that will be used when TTree::Draw() commands are issued) and an\n\
alias. The visible name is the alias. Aliases of user defined expressions\n\
have a leading ~ and may be used in new expressions. Expressions containing\n\
boolean operators have a specific icon and may be dragged to the active cut\n\
(scissors item) position.\n\n\
The expression editor can be activated by double-clicking empty expression,\n\
using <EditExpression> from the selected expression context menu or using\n\
the <Edit/Expression> menu.\n\n\
The editor will pop-up in the left part, but it can be moved.\n\
The editor usage is the following:\n\
  - you can write C expressions made of leaf names by hand or you can insert\n\
    any item from the right panel by clicking on it (recommandable)\n\
  - you can click on other expressions/leaves to paste them in the editor\n\
  - you should write the item alias by hand since it not makes the expression\n\
    meaningfull and also highly improves the layout for big expressions\n\
  - you may redefine an old alias - the other expressions depending on it will\n\
    be modified accordingly. Must NOT be the leading string of other aliases.\n\
    When Draw commands are issued, the name of the corresponding histogram axes\n\
    will become the aliases of the expressions.\n\n\
";

const char gTVHelpSession[] ="\
A TreeViewer session is made by the list of user-defined expressions and cuts,\n\
applying to a specified tree. A session can be saved using File/SaveSource menu\n\
or the SaveSource method from the context menu of the right panel. This will\n\
create a macro having as default name treeviewer.C that can be ran at any time to\n\
reproduce the session.\n\n\
Besides the list of user-defined expressions, a session may contain a list of\n\
RECORDS. A record can be produced in the following way:\n\
   - dragging leaves/expression on X/Y/Z\n\
   - changing drawing options\n\
   - clicking the RED button on the bottom when happy with the histogram\n\
NOTE that just double clicking a leaf will not produce a record: the histogram must\n\
be produced when clicking the DRAW button on the bottom-left.\n\n\
The records will appear on the list of records in the bottom right of the\n\
tree viewer. Selecting a record will draw the corresponding histogram. Records\n\
can be played using the arrow buttons near to the record button. When saving the\n\
session, the list of records is being saved as well.\n\n\
Records have a default name corresponding to the Z:Y:X selection, but this can be\n\
changed using SetRecordName() method from the right panel context menu.\n\n\
";

const char gTVHelpUserCommands[] = "\
User commands can be issued directly from the textbox labeled <Command>\n\
from the upper-left toolbar by typing and pressing Enter at the end.\n\
Another way is from the right panel context menu: ExecuteCommand.\n\n\
All commands can be interrupted at any time by pressing the STOP button\n\
from the bottom-left.\n\n\
You can toggle recording of the current command in the history file by\n\
checking the Rec button from the top-right.\n\n\
";

const char gTVHelpContext[] = "\
You can activate context menus by right-clicking on items or inside the\n\
right panel.\n\n\
Context menus for mapped items from the left tree-type list:\n\n\
The items from the left that are provided with context menus are tree and\n\
branch items. You can directly activate the *MENU* marked methods of TTree\n\
from this menu.\n\n\
Context menu for the right panel:\n\n\
  A general context menu is acivated if the user right-clicks the right panel.\n\
  Commands are:\n\
  - EmptyAll        : clears the content of all expressions\n\
  - ExecuteCommand  : execute a ROOT command\n\
  - MakeSelector    : equivalent of TTree::MakeSelector()\n\
  - NewExpression   : add an expression item in the right panel\n\
  - Process         : equivalent of TTree::Process()\n\
  - SaveSource      : save the current session as a C++ macro\n\
  - SetScanFileName : define a name for the file where TTree::Scan command\n\
    is redirected when the <Scan> button is checked\n\
  - SetTreeName     : open a new tree whith this name in the viewer.\n\n\
    A specific context menu is activated if expressions/leaves are\n\
    right-clicked.\n\
  Commands are:\n\
  - Draw            : draw a histogram for this item\n\
  - EditExpression  : pops-up the expression editor\n\
  - Empty           : empty the name and alias of this item\n\
  - RemoveItem      : removes clicked item from the list\n\
  - Scan            : scan this expression\n\
  - SetExpression   : edit name and alias for this item by hand\n\n\
";

const char gTVHelpDrawing[] = "\
Fast variable drawing: just double-click an item from the right list.\n\n\
Normal drawing: Edit the X, Y, Z fields or drag expressions here, fill\n\
input-output list names if you have defined them. Press <Draw> button.\n\n\
You can change output histogram name, or toggle Hist or Scan modes by checking\n\
the corresponding buttons.\n\
Hist mode implies that the current histogram will be redrawn with the current\n\
graphics options. If a histogram is already drawn and the graphic <Option>\n\
is changed, pressing <Enter> will redraw with the new option. Checking <Scan>\n\
will redirect TTree::Scan() command in an ASCII file.\n\n\
You have a complete list of histogram options in multicheckable lists from\n\
the Option menu. When using this menu, only the options compatible with the\n\
current histogram dimension will be available. You can check multiple options\n\
and reset them by checking the Default entries.\n\n\
After completing the previous operations you can issue the draw command by\n\
pressing the DRAW button.\n\n\
";

const char gTVHelpMacros[] = "\
Macros can be loaded and executed in this version only by issuing\n\
the corresponding user commands (see help on user commands).\n\n\
";

