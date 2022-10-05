#pragma once

#include <TBrowser.h>

class RooNode;

class RooBrowser: public TBrowser {
public:
    RooBrowser();
    RooBrowser(RooNode* o);

    ~RooBrowser();

    RooNode* GetSelected();

    void ls(const char* path = nullptr);

    void cd(const char* path);

private:
    std::shared_ptr<RooNode> fNode; //!

    std::shared_ptr<RooNode> fTopNode; //!

    ClassDefOverride(RooBrowser, 0);
};
