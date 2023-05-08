/*
 * Project: RooFit
 * Author:
 *   Will Buttinger, RAL 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class ROOT::Experimental::RooBrowser
\ingroup RooFit

 \image html RooBrowser.png width=50%

To get started with the RooBrowser, open any ROOT file containing a workspace
 and then create an instance of the `ROOT::Experimental::RooBrowser` just like
 creating an instance of a `TBrowser`. A window will be displayed with a navigable
 tree structure on the left that lets you explore the content of the workspaces
 present in the loaded file. Note that additional files, <b>including json workspace files</b>,
 can be loaded through the `Browser --> Open` menu in the top left corner.

The context menu for each node (access by right clicking on the node) in the tree structure can be used to get more
information about the node. In particular, the `Draw` command can be selected on many of the nodes that are part of a
statistical model, which will visualize that part of the model in the browser window. A number of options are available
for the `Draw` command, including (some options can be combined):

 - "e" : calculate and visualize propagated model uncertainty
 - "auxratio" : Draw a ratio auxiliary plot below the main plot
 - "auxsignif" : Draw a significance auxiliary plot below the main plot
 - "pull" : show panel of current parameter values, which can be dragged in order to change the values and visualize the
effect on the model (very experimental feature).

 Once a node has been drawn, the styling of subsequent draws can be controlled through `TStyle` objects
 that will now appear in the `objects` folder in the workspace.

A model can be fit to a dataset from the workspace using the `fitTo` context menu command and specifying
 the name of a dataset in the workspace (if no name is given, an expected dataset corresponding to the
 current state of the model will be used). A dialog will display the fit result status code when the
 fit completes and then a `fits` folder will be found under the workspace (the workspace may need to
 be collapsed and re-expanded to make it appear) where the fit result can be found, selected, and visualized.
 In multi-channel models the channels that are included in the fit can be controlled with the checkboxes
 in the browser. Clicking the checkbox will cycle through three states: checked, unchecked with
 grey-underline, and checked with grey-underline. The grey-underline indicates that channel wont be
 included in the fit (and will appear greyed out when the model is visualized)

Many more features are available in the `RooBrowser`, and further documentation and development can be found at
 the <a href="https://gitlab.cern.ch/will/xroofit">xRooFit</a> repository, which is the library where the browser has
 been originally developed. The author (Will Buttinger) is also very happy to be contacted with questions or
 feedback about this new functionality.

 */

#ifndef RooFit_RooBrowser_h
#define RooFit_RooBrowser_h

#include "RooFit/xRooFit/xRooBrowser.h"

namespace ROOT::Experimental {
using RooBrowser = XROOFIT_NAMESPACE::xRooBrowser;
}

#endif
