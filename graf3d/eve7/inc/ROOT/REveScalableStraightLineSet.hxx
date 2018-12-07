#ifndef ROOT_REveScalableStraightLineSet
#define ROOT_REveScalableStraightLineSet

#include "ROOT/REveStraightLineSet.hxx"

namespace ROOT {
namespace Experimental {
class REveScalableStraightLineSet : public REveStraightLineSet
{
private:
   REveScalableStraightLineSet(const REveScalableStraightLineSet&);            // Not implemented
   REveScalableStraightLineSet& operator=(const REveScalableStraightLineSet&); // Not implemented

protected:
   Double_t      fCurrentScale;
   Float_t       fScaleCenter[3];

public:
   REveScalableStraightLineSet(const char* n="ScalableStraightLineSet", const char* t="");
   virtual ~REveScalableStraightLineSet() {}

   void SetScaleCenter(Float_t x, Float_t y, Float_t z);
   void SetScale(Double_t scale);

   Double_t GetScale() const;
};

} // namespace Experimental
} // namespace ROOT
#endif
