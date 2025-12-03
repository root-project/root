#ifndef CLASSES
#define CLASSES

#include <vector>
#include "TObject.h"
#include "Math/GenVector/Cartesian3D.h"


// template <class T>
// class Vertex{
// public:
//    Vertex(T x, T y, T z):m_x(x),m_y(y),m_z(z){};
//    T X(){return m_x;};
//    T Y(){return m_y;};
//    T Z(){return m_z;};
// private:
//    T m_x, m_y, m_z;   
// };


typedef ROOT::Math::Cartesian3D<Double32_t> Vertex32;
typedef ROOT::Math::Cartesian3D<double> Vertex;

class SuperCluster{
public:   
   SuperCluster():energy(-1){};
   double GetEnergy() const {return energy;};
   void SetEnergy(double en) {energy=en;};
private:
   double energy;
};
class Particle {
public:
   Particle():vertex(.1,.2,.3){};
   const Vertex& GetVertex() const {return vertex;};
private:
   Vertex vertex;
};


class Container: public TObject {
public:
   Container(){};
   const SuperCluster& GetSc() const {return m_sc;};
   const Particle& GetParticle() const {return m_part;};
   void SetScEnergy(double energy) {m_sc.SetEnergy(energy);}
private:
   SuperCluster m_sc;
   Particle m_part;

ClassDefOverride(Container,1);
};

#endif
