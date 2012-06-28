/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// t1030.h 
// Outrageously long template class names
#include <stdio.h>

#include <vector>
#include <map>
#include <string>

using namespace std;

namespace Experiment1 {

  class TExperimentalDataNode {
  public:
    ~TExperimentalDataNode() { printf("TExperimentalDataNode\n"); }
    TExperimentalDataNode() { printf("~TExperimentalDataNode\n"); }
    TExperimentalDataNode(const TExperimentalDataNode& x) { printf("TExperimentalDataNode(const TExperimentalDataNode&)\n"); }
    
    TExperimentalDataNode& operator=(const TExperimentalDataNode& x) { return(*this); }
  };
  
  class TExperimentalData {
  public:
    vector<TExperimentalDataNode> m_d;
    TExperimentalData() { printf("TExperimentalData\n"); }
    ~TExperimentalData() {printf("~TExperimentalData\n");  }
    TExperimentalData(const TExperimentalData& x) { printf("TExperimentalData(const TExperimentalData&)\n"); }
    
    const TExperimentalData& operator=(const TExperimentalData& x) { return(*this); }
    void SetNode(const TExperimentalDataNode& x) { }
  };
  
}

bool operator==(const Experiment1::TExperimentalDataNode& a,const Experiment1::TExperimentalDataNode& x) { return(true); }
bool operator!=(const Experiment1::TExperimentalDataNode& a,const Experiment1::TExperimentalDataNode& x) { return(true); }
bool operator>(const Experiment1::TExperimentalDataNode& a,const Experiment1::TExperimentalDataNode& x) { return(true); }
bool operator<(const Experiment1::TExperimentalDataNode& a,const Experiment1::TExperimentalDataNode& x) { return(true); }
bool operator>=(const Experiment1::TExperimentalDataNode& a,const Experiment1::TExperimentalDataNode& x) { return(true); }
bool operator<=(const Experiment1::TExperimentalDataNode& a,const Experiment1::TExperimentalDataNode& x) { return(true); }

bool operator==(const Experiment1::TExperimentalData& a,const Experiment1::TExperimentalData& x) { return(true); }
bool operator!=(const Experiment1::TExperimentalData& a,const Experiment1::TExperimentalData& x) { return(true); }
bool operator>(const Experiment1::TExperimentalData& a,const Experiment1::TExperimentalData& x) { return(true); }
bool operator<(const Experiment1::TExperimentalData& a,const Experiment1::TExperimentalData& x) { return(true); }
bool operator>=(const Experiment1::TExperimentalData& a,const Experiment1::TExperimentalData& x) { return(true); }
bool operator<=(const Experiment1::TExperimentalData& a,const Experiment1::TExperimentalData& x) { return(true); }

#ifdef __MAKECINT__

#pragma link C++ class Experiment1::TExperimentalDataNode;
#pragma link C++ class Experiment1::TExperimentalData;

#pragma link C++ class vector<Experiment1::TExperimentalData>;
#pragma link C++ class vector<vector<Experiment1::TExperimentalData> >;
#pragma link C++ class vector<vector<vector<Experiment1::TExperimentalData> > >;

#pragma link C++ class map<string,Experiment1::TExperimentalData>;
#pragma link C++ class map<string,vector<Experiment1::TExperimentalData> >;
#pragma link C++ class map<string,vector<vector<Experiment1::TExperimentalData> > >;

#if 0
// With following very complicated STL container, Cint has fixed length
// string buffer limitation of G__LONGLINE
//  ifunc.c G__readansiproto() paraname[G__LONGLINE] 
#pragma link C++ class map<string,vector<vector<vector<Experiment1::TExperimentalData> > > >;
#endif

#endif


