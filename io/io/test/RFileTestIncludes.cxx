#include "RFileTestIncludes.hxx"

TTreeDestructorCounter::~TTreeDestructorCounter() {
  ++fgTimesDestructed;
}

void TTreeDestructorCounter::ResetTimesDestructed()
{
   fgTimesDestructed = 0;
}

int TTreeDestructorCounter::GetTimesDestructed()
{
   return fgTimesDestructed;
}

int TTreeDestructorCounter::fgTimesDestructed = 0;
