#pragma once

#include "xRooNode.h"

#include "TBrowser.h"

#ifdef XROOFIT_NAMESPACE
namespace XROOFIT_NAMESPACE {
#endif

class xRooBrowser: public TBrowser {
public:
    xRooBrowser();
    xRooBrowser(xRooNode* o);

    xRooNode* GetSelected() { return dynamic_cast<xRooNode*>(TBrowser::GetSelected()); }

    void ls(const char* path = nullptr) const override {
        if (!fNode) return;
        if (!path) fNode->Print();
        else {
            // will throw exception if not found
            fNode->at(path)->Print();
        }
    }

    void cd(const char* path) {
        auto _node = fNode->at(path); // throws exception if not found
        fNode = _node;
    }

private:
    std::shared_ptr<xRooNode> fNode; //!

    std::shared_ptr<xRooNode> fTopNode; //!

ClassDefOverride(TBrowser,0)

};

#ifdef XROOFIT_NAMESPACE
}
#endif