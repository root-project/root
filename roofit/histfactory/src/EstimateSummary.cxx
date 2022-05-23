// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////

/** \class RooStats::HistFactory::EstimateSummary
 *  \ingroup HistFactory
 */

#include <algorithm>
#include "RooStats/HistFactory/EstimateSummary.h"

ClassImp(RooStats::HistFactory::EstimateSummary);

using namespace std;

namespace RooStats {
  namespace HistFactory {

    EstimateSummary::EstimateSummary(){
      nominal=0;

      normName="Lumi";
      IncludeStatError = false;
      StatConstraintType=Gaussian;
      RelErrorThreshold=0.0;
      relStatError=nullptr;
      shapeFactorName="";
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

    void EstimateSummary::AddSyst(const  string &sname, TH1* low, TH1* high){
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
      if(! (shapeFactorName==other.shapeFactorName)){
        cout << "norm names don't match : " << shapeFactorName << " vs " << other.shapeFactorName << endl;
        return false;
      }
      if (nominal && other.nominal)
      if(! CompareHisto( this->nominal,  other.nominal ) ) {
        cout << "nominal histo don't match" << endl;
        return false;
      }
      if(! (IncludeStatError==other.IncludeStatError)){
        cout << "Include Stat Error bools don't match : " << IncludeStatError << " vs " << other.IncludeStatError << endl;
        return false;
      }
      if(! (StatConstraintType==other.StatConstraintType)){
        cout << "Stat Constraint Types don't match : " << StatConstraintType << " vs " << other.StatConstraintType << endl;
        return false;
      }
      if(! (RelErrorThreshold==other.RelErrorThreshold)){
        cout << "Relative Stat Error Thresholds don't match : " << RelErrorThreshold << " vs " << other.RelErrorThreshold << endl;
        return false;
      }
      if (relStatError && other.relStatError)
      if(! CompareHisto( this->relStatError,  other.relStatError ) ) {
        cout << "relStatError histo don't match" << endl;
        return false;
      }
      if(! (shapeFactorName==other.shapeFactorName)){
        cout << "Shape Factor Names don't match : " << shapeFactorName << " vs " << other.shapeFactorName << endl;
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

