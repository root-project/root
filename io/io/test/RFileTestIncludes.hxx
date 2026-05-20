#ifndef ROOT_RFILE_TEST_INCLUDES
#define ROOT_RFILE_TEST_INCLUDES

#include <TTree.h>

// WARNING: this class is used in some tests that check its `fgTimesDestructed`.
// ResetTimesDestructed() must be called before using this class for consistency.
class TTreeDestructorCounter : public TTree {
   static int fgTimesDestructed;

public:
   static void ResetTimesDestructed();
   static int GetTimesDestructed();

   using TTree::TTree;
   ~TTreeDestructorCounter();

   ClassDefOverride(TTreeDestructorCounter, 2);
};

#endif
