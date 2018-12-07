#ifndef ROOT7_REveViewContext
#define ROOT7_REveViewContext

namespace ROOT {
namespace Experimental {

class REveTableViewInfo;
class REveTrackPropagator;

class REveViewContext  {
private:
   float m_R;
   float m_Z;
   REveTrackPropagator* m_trackPropagator;

   REveTableViewInfo* fTableInfo;

public:
   REveViewContext(): m_R(100), m_Z(100), m_trackPropagator(0), fTableInfo(0) {}
   virtual ~REveViewContext(){}

   void SetBarrel(float r, float z) { m_R = r; m_Z = z; }
   void SetTrackPropagator( REveTrackPropagator* p) {m_trackPropagator = p; }
   void SetTableViewInfo(REveTableViewInfo* ti) { fTableInfo = ti; }

   float GetMaxR() const {return m_R;}
   float GetMaxZ() const {return m_Z;}
   REveTrackPropagator* GetPropagator() const {return m_trackPropagator;}
   REveTableViewInfo* GetTableViewInfo() const {return fTableInfo;}

   ClassDef(REveViewContext, 0);
};
}
}

#endif
