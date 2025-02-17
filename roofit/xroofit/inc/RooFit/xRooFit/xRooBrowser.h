/*
 * Project: xRooFit
 * Author:
 *   Will Buttinger, RAL 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "Config.h"

#ifdef XROOFIT_USE_PRAGMA_ONCE
#pragma once
#endif
#if !defined(XROOFIT_XROOBROWSER_H) || defined(XROOFIT_USE_PRAGMA_ONCE)
#ifndef XROOFIT_USE_PRAGMA_ONCE
#define XROOFIT_XROOBROWSER_H
#endif

#include "TBrowser.h"
#include "TQObject.h"

BEGIN_XROOFIT_NAMESPACE

class xRooNode;

class xRooBrowser : public TBrowser, public TQObject {
public:
   xRooBrowser();
   xRooBrowser(xRooNode *o);

   xRooNode *GetSelected();

   xRooNode *Open(const char *filename);

   void ls(const char *path = nullptr) const override;
   void cd(const char *path);

   void HandleMenu(Int_t id);

private:
   std::shared_ptr<xRooNode> fNode;    //!
   std::shared_ptr<xRooNode> fTopNode; //!

public:
   ClassDefOverride(xRooBrowser, 0)
};

END_XROOFIT_NAMESPACE

#endif // include guard
