// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
BEGIN_HTML
<p>
</p>
END_HTML
*/
//

#include <algorithm>
#include "RooStats/HistFactory/EstimateSummary.h"

ClassImp(RooStats::HistFactory::EstimateSummary)

using namespace std; 

namespace RooStats {
  namespace HistFactory {

    EstimateSummary::EstimateSummary(){
      nominal=0; 
      normName="Lumi";
    }
    EstimateSummary::~EstimateSummary(){}

    void EstimateSummary::Print(const char * /*opt*/) const {
      cout << "EstimateSummary (name = " << name << " empty = " << name.empty() << ")"<< endl;
      cout << "  TObj name = " << this->GetName() << endl;
      cout << "  Channel = " << channel << endl;
      cout << "  NormName = " << normName << endl;
      cout << "  Nominal ptr = " << nominal << endl;
      if (nominal) cout << "  Nominal hist name = " << nominal->GetName() << endl;
      cout << "  Number of hist variations = " << systSourceForHist.size() 
     << " " << lowHists.size() << " " 
     << " " << highHists.size() << endl;
      cout << "  Number of overall systematics = " << overallSyst.size() << endl;
    }

    void EstimateSummary::AddSyst(const  string &sname, TH1F* low, TH1F* high){
      systSourceForHist.push_back(sname);
      lowHists.push_back(low);
      highHists.push_back(high);
    }

    bool EstimateSummary::operator==(const EstimateSummary &other) const {
      // Comparator for two Estimate summary objects. Useful to make sure two analyses are the same

      //this->print();
      //other.print();
      if(! (name==other.name)){
        cout << "names don't match : " << name << " vs " << other.name << endl;
        return false;
      }
      if(! (channel==other.channel)){
        cout << "channel names don't match : " << channel << " vs " << other.channel << endl;
        return false;
      }
      if(! (normName==other.normName)){
        cout << "norm names don't match : " << normName << " vs " << other.normName << endl;
        return false;
      }
      if (nominal && other.nominal)
      if(! CompareHisto( this->nominal,  other.nominal ) ) {
        cout << "nominal histo don't match" << endl;
        return false;
      }
      /// compare histo sys
      int counter=0;
      for( vector<string>::const_iterator itr=systSourceForHist.begin(); itr!=systSourceForHist.end(); ++itr){
        unsigned int ind = find(other.systSourceForHist.begin(), other.systSourceForHist.end(), *itr) - other.systSourceForHist.begin();
        if(ind<other.systSourceForHist.size() && systSourceForHist.size() == other.systSourceForHist.size()){
          if(! (CompareHisto( lowHists[ counter ], other.lowHists[ ind ]))){
            cout << "contents of sys histo low " << *itr << " did not match" << endl;
          }
          else if (!( CompareHisto( highHists[counter], other.highHists[ ind ]) ) ){
            cout << "contents of sys histo high " << *itr << " did not match" << endl;
          } 
        } else {
          cout << "mismatch in systSourceForHist : " << systSourceForHist.size() << " vs " << other.systSourceForHist.size() << endl;
          for( vector<string>::const_iterator itr_this=systSourceForHist.begin(); itr_this!=systSourceForHist.end(); ++itr_this){
            cout << "  this contains: " << *itr_this << endl;
          }
          for( vector<string>::const_iterator itr_other=other.systSourceForHist.begin(); itr_other!=other.systSourceForHist.end(); ++itr_other){
            cout << "  other contains: " << *itr_other << endl;
          }
          return false;
        }
        counter++;
      }
      /// compare overall sys
      if( overallSyst.size() != other.overallSyst.size()){
        cout << "mismatch in overallSyst : " << overallSyst.size() << " vs " << other.overallSyst.size() << endl;
        return false;
      }
      for( map<string, pair<double, double> >::const_iterator itr=overallSyst.begin(); itr!=overallSyst.end(); ++itr){
        map<string, pair<double, double> >::const_iterator found=other.overallSyst.find(itr->first);
        if(found==other.overallSyst.end()){
          cout << "mismatch in overallSyst, didn't find " << itr->first << endl;
          return false;
        }
        if(! (itr->second.first==found->second.first && itr->second.second==found->second.second)){
          cout << "mismatch in overall Syst value of " << itr->first << endl;
          return false;
        }
      }
      return true;
    }

    bool EstimateSummary::CompareHisto( const TH1 * one, const TH1 * two) const {

       if (!one && !two) return true; 
       if (!one) return false; 
       if (!two) return false; 
      
      for(int i=1; i<=one->GetNbinsX(); ++i){
        if(!(one->GetBinContent(i)-two->GetBinContent(i)==0)) return false;
      }
      return true;
      //if(one->Integral()-two->Integral()==0) return true;
      //cout << "Integral of " << one->GetName() <<  " : " << one->Integral() << " vs Integral ov " << two->GetName() << " : " << two->Integral() << endl;
    }

  }
}

