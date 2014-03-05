#ifndef CLASSES
#define CLASSES

#include "Math/GenVector/PositionVector3D.h"
#include <vector>
#include "TObject.h"

typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag> Vertex;

class SuperCluster {
public:   
   SuperCluster():energy(123){};
   double GetEnergy() const {return energy;};
private:
   double energy;
};
class Particle {
public:
   Particle():vertex(.1,.2,.3){};
   const Vertex& GetVertex() const {return vertex;};
private:
   ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag> vertex;
};


class Container: public TObject {
public:
   Container(){};
   const SuperCluster& GetSc() const {return m_sc;};
   const Particle& GetParticle() const {return m_part;};
private:
   SuperCluster m_sc;
   Particle m_part;

ClassDef(Container,1);
};

#endif
