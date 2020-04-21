/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDrawableRequest.hxx"

#include "ROOT/RDrawable.hxx"
#include "ROOT/RCanvas.hxx"

using namespace ROOT::Experimental;


/////////////////////////////////////////////////////////////////////////////
/// destructor, pin vtable

RDrawableReply::~RDrawableReply() = default;

/////////////////////////////////////////////////////////////////////////////
/// destructor, pin vtable

RDrawableRequest::~RDrawableRequest() = default;


/////////////////////////////////////////////////////////////////////////////
/// Execute method of the drawable

std::unique_ptr<RDrawableReply> RDrawableExecRequest::Process()
{
   if (!exec.empty() && GetContext().GetDrawable())
      GetContext().GetDrawable()->Execute(exec);

   if (GetContext().GetCanvas())
      GetContext().GetCanvas()->Modified();

   return nullptr;
}
