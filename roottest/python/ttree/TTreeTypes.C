/*
  File: roottest/python/ttree/TTreeTypes.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 11/17/06
  Last: 12/23/13
*/

#include <algorithm>
#include <vector>

#include "TH1D.h"
#include "TTree.h"

#ifdef __MAKECINT__
#pragma link C++ class std::vector< vector< float > >;
#pragma link C++ class std::vector< vector< float > >::iterator;
#pragma link C++ class std::vector< vector< float > >::const_iterator;
#endif


typedef std::vector< float >                 Floats_t;
typedef std::vector< std::vector< float > >  Tuples_t;


class SomeDataObject {
public:
   const Floats_t& GetFloats() { return m_floats; }
   const Tuples_t& GetTuples() { return m_tuples; }

public:
   void AddFloat( float f ) { m_floats.push_back( f ); }
   void AddTuple( const std::vector< float >& t ) { m_tuples.push_back( t ); }

private:
   Floats_t m_floats;
   Tuples_t m_tuples;
};


struct SomeDataStruct {
   Floats_t Floats;
   Char_t   Label[3];
   Int_t    NLabel;
};


struct Output_Vars {
   Double_t t0[28];
};

void CreateArrayTree() {
   static Output_Vars output_vars;

   TTree* outtree = new TTree( "Proto2Analyzed","Proto2 Data with First-Level Analysis" );
   outtree->Branch( "output_vars", &output_vars, 32000, 1 );

   std::fill_n( output_vars.t0, 28, 0 );

   double vals[] = { -1 , -1 , 428 , 0 , -1 ,
                    167 , 0 , 0 , 0 , 403 ,
                    -1 , -1 , 270 , -1 , 0 ,
                     -1 , 408 , 0 , -1 , 198 };
   for ( int i = 0; i < (int)(sizeof(vals)/sizeof(vals[0])); ++i ) {
      output_vars.t0[i] = vals[i];
   }

   outtree->Fill();
}
