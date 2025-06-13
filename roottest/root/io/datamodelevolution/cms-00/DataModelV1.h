//--------------------------------------------------------------------*- C++ -*-
// file:   DataModelV1.h
// author: Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef DataModelV1_h
#define DataModelV1_h

#include <utility>
#include <vector>

namespace CMS {
   class ClassAIns
   {
   public:
      struct Transient {
         Transient() : fCached(false) {}
         bool fCached; //!
      };

      ClassAIns(): m_a( 12 ), m_b( 32.23 ) {}
   private:
      int    m_a;
      double m_b;
   };
}

class ClassAIns
{
public:
   struct Transient {
      Transient() : fCached(false) {}
      bool fCached; //!
   };

   ClassAIns(): m_a( 12 ), m_b( 32.23 ) {}
private:
   int    m_a;
   double m_b;
   ClassAIns::Transient m_cache; //!
};

class ClassABase
{
   public:
      ClassABase(): m_a( 3 ), m_b( 123.33 ) {}
      virtual ~ClassABase() {};
   private:
      int    m_a;
      double m_b;
};

class ClassA: public ClassABase
{
   public:
      ClassA(): m_c( 65.22 ), m_e( 8 ), m_unit(1) {}
      virtual ~ClassA() {}
   private:
      double    m_c;
      ClassAIns m_d;
      unsigned int  m_e;
      int       m_unit;
};

class ClassB: public ClassA
{
   public:
      ClassB(): m_f( 34 ), m_g( 12.22 ) {}
      virtual ~ClassB() {};
   private:
      int    m_f;
      double m_g;
};

class ClassC: public ClassABase
{
   public:
      ClassC(): m_f( 74.22 ), m_g( 199.22 ) {}
      virtual ~ClassC() {};
   private:
      double m_f;
      double m_g;
//      std::vector<std::pair<double, int> > m_h;
};

class ClassD
   {
   public:
      ClassD(): m_c( 65.22 ), m_e( 8 ) {}
      virtual ~ClassD() {}
   private:
      float      m_c;
      ClassAIns  m_d;
      int        m_e;
   };

struct _dummy
{
   std::vector<double>                  a1;
   std::pair<int,double>                a2;
   std::vector<std::pair<int, double> > a3;
   std::vector<ClassA>                  a4;
   std::vector<ClassA*>                 a5;
   std::vector<ClassB>                  a6;
   std::vector<ClassB*>                 a7;
   std::vector<ClassC>                  a8;
   std::vector<ClassC*>                 a9;
   std::vector<ClassD>                  a10;
   std::vector<ClassD*>                 a11;
};

// LHCb

#include "Math/SMatrix.h"

namespace Gaudi {
   typedef ROOT::Math::SMatrix<double, 1, 5> TrackProjectionMatrix;
   typedef ROOT::Math::SMatrix<double, 2, 5> TrackProjectionMatrixMis;
}

namespace LHCb {

   class RefLeft {
   private:
      long fValue;
   public:
      void Set(long v) { fValue = v; }
      long GetValue() { return fValue; }
      RefLeft() : fValue(0) {}

   };

   class RefRight {
   private:
      long fValue;
   public:
      void Set(long v) { fValue = v; }
      long GetValue() { return fValue; };
      RefRight() : fValue(0) {}

   };

   class Ref {
   private:
      long fLeft;
      long fRight;
   public:
      void SetLeft(long v) { fLeft = v; }
      void SetRight(long v) { fRight = v; }
      Ref() : fLeft(0),fRight(0) {}

   };


// The interesting part is LHCb::Node:
class Node  {
private:
   double                          m_type;             ///< type of node
   //LHCb::State                   m_state;            ///< state
   //LHCb::StateVector             m_refVector;        ///< the reference vector
   bool                            m_refIsSet;         ///< flag for the reference vector
   //LHCb::Measurement*            m_measurement;      ///< pointer to the measurement (not owner)
   double                          m_residual;         ///< the residual value
   double                          m_errResidual;      ///< the residual error
   double                          m_errMeasure;       ///< the measure error
   Gaudi::TrackProjectionMatrix    m_projectionMatrix; ///< the projection matrix
   Gaudi::TrackProjectionMatrixMis m_projection_file;  ///< the projection matrix
};

class Track /* : public KeyedObject<int> */  {
public:
   Track() { m_nodes.push_back(new LHCb::Node()); }
   ~Track() { for(unsigned int i=0; i<m_nodes.size(); ++i) { delete m_nodes[i]; }; m_nodes.clear(); }
   void SetRef(long l, long r) { fLeft.Set(l); fRight.Set(r); }
private:
   double                            m_chi2PerDoF;           ///< Chi2 per degree of freedom of the track
   int                               m_nDoF;                 ///< Number of degrees of freedom of the track
   double                            m_likelihood;           ///< Likelihood variable
   double                            m_ghostProbability;     ///< ghost probability variable
   unsigned int                      m_flags;                ///< The variety of track flags
   //std::vector<LHCb::LHCbID>       m_lhcbIDs;              ///< Container of (sorted) LHCbIDs
   //std::vector<LHCb::State*>       m_states;               ///< Container with pointers to all the states
   //std::vector<LHCb::Measurement*> m_measurements;         ///< Container of Measurements
   //Gaudi::TrackProjectionMatrix    m_projections;      //! ///< the projection matrix
   Node                              m_nothing;          //! ///< Just for testing
   std::vector<LHCb::Node*>          m_nodes;            //! ///< Container of Nodes
   Gaudi::TrackProjectionMatrixMis   m_projections_file;     ///< the projection matrix
   //ExtraInfo                       m_extraInfo;            ///< Additional pattern recognition information. Do not access directly, use *Info() methods instead.
   //SmartRefVector<LHCb::Track>     m_ancestors;            ///< Ancestor tracks that created this one

   RefLeft  fLeft;
   RefRight fRight;

};
}

#endif
