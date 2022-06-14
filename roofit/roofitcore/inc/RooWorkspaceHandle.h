// Author: Stephan Hageboeck, CERN  01/2019

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOFIT_ROOFITCORE_INC_ROOWORKSPACEHANDLE_H_
#define ROOFIT_ROOFITCORE_INC_ROOWORKSPACEHANDLE_H_

#include "RooWorkspace.h"

/// An interface to set and retrieve a workspace.
/// This is needed for all generic objects that can be saved in a workspace, which itself depend
/// on the workspace (e.g. the RooStats::ModelConfig).
/// Because of a circular dependency, a workspace with a ModelConfig cannot be (deep) cloned.
/// The handle hides this dependency.
class RooWorkspaceHandle {
public:
   virtual ~RooWorkspaceHandle() {}

   ///Set the workspace. If it exists, it is up to the implementing class to decide how to proceed.
   virtual void SetWS(RooWorkspace &ws) = 0;

   ///Set the workspace irrespective of what the previous workspace is.
   virtual void ReplaceWS(RooWorkspace *ws) = 0;

   ///Retrieve the workspace.
   virtual RooWorkspace *GetWS() const = 0;

   ClassDef(RooWorkspaceHandle, 0)
};

#endif /* ROOFIT_ROOFITCORE_INC_ROOWORKSPACEHANDLE_H_ */
