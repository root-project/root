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

#ifndef XROOFIT_NAMESPACE
#pragma once
#endif
#if !defined(XROOFIT_XROOBROWSER_H) || !defined(XROOFIT_NAMESPACE)
#define XROOFIT_XROOBROWSER_H

#include "TBrowser.h"

#ifdef XROOFIT_NAMESPACE
namespace XROOFIT_NAMESPACE {
#endif

class xRooNode;

class xRooBrowser: public TBrowser {
public:
    xRooBrowser();
    xRooBrowser(xRooNode* o);

    xRooNode* GetSelected();

    void ls(const char* path = nullptr) const override;
    void cd(const char* path);

private:
    std::shared_ptr<xRooNode> fNode; //!
    std::shared_ptr<xRooNode> fTopNode; //!

ClassDefOverride(TBrowser,0)

};

#ifdef XROOFIT_NAMESPACE
}
#endif

#endif // include guard