#ifndef __Track_TObject__
#define __Track_TObject__

#include "TObject.h"

class Track : public TObject {
public:
  std::vector<double> hitEnergies;
  int id = 42;

  Track(const std::vector<double> &v = {}) : hitEnergies{v} {};

  ClassDefOverride(Track, 1);
};

#endif
