#ifndef ROOT_REveScalableStraightLineSet
#define ROOT_REveScalableStraightLineSet

#include "ROOT/REveStraightLineSet.hxx"

namespace ROOT {
namespace Experimental {

class REveScalableStraightLineSet : public REveStraightLineSet
{
private:
   REveScalableStraightLineSet(const REveScalableStraightLineSet&) = delete;
   REveScalableStraightLineSet& operator=(const REveScalableStraightLineSet&) = delete;

protected:
   Double_t      fCurrentScale;
   Float_t       fScaleCenter[3];

public:
   REveScalableStraightLineSet(const std::string &n = "ScalableStraightLineSet", const std::string &t = "");
   virtual ~REveScalableStraightLineSet() {}

   void SetScaleCenter(Float_t x, Float_t y, Float_t z);
   void SetScale(Double_t scale);

   Double_t GetScale() const;
};

} // namespace Experimental
} // namespace ROOT
#endif
