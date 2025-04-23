#ifndef H_ODIN
#define H_ODIN

// A mock of the LHCb::ODIN class part of LHCb

static const unsigned int CLID_ODIN = 14071789;

class ODIN{
public:
   static const unsigned int& classID() {return CLID_ODIN;};
};
#endif