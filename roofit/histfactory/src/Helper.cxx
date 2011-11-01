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



//#define DEBUG
#include "Helper.h"

namespace RooStats{
namespace HistFactory{
  vector<pair<string, string> > get_comb(vector<string> names){
    vector<pair<string, string> > list;
    for(vector<string>::iterator itr=names.begin(); itr!=names.end(); ++itr){
      vector<string>::iterator itr2=itr; 
      for(itr2++; itr2!=names.end(); ++itr2){
        list.push_back(pair<string, string>(*itr, *itr2));
      }
    }
    return list;
  }

  vector<EstimateSummary>*  loadSavedInputs(TFile* outFile, string channel ){
    outFile->ShowStreamerInfo();

    vector<EstimateSummary>* summaries = new  vector<EstimateSummary>;
    outFile->cd(channel.c_str());

    // loop through estimate summaries
    TIter next(gDirectory->GetListOfKeys()); 
    EstimateSummary* summary; 
    while ((summary=(EstimateSummary*) next())) { 
//      if(summary){
        summary->Print();
        cout << "was able to read summary with name " << summary->name << endl;
        cout << " nominal hist = " << summary->nominal << endl;
        if(summary->nominal)
           cout << " hist name = " << summary->nominal->GetName() <<endl;
        cout << "still ok" << endl;
       
        summaries->push_back(*summary);

//L.M. This code cannot be reached- remove it 
//       }
//       else{
//         cout << "was not able to read summary" << endl;
//       }
    } 
    return summaries;
  }


  void saveInputs(TFile* outFile, string channel, vector<EstimateSummary> summaries){
    vector<EstimateSummary>::iterator it = summaries.begin();
    vector<EstimateSummary>::iterator end = summaries.end();
    vector<TH1*>::iterator histIt;
    vector<TH1*>::iterator histEnd;
    outFile->mkdir(channel.c_str());

    for(; it!=end; ++it){
      if(channel != it->channel){
        cout << "channel mismatch" << endl;
        return;
      }
      outFile->cd(channel.c_str());
      
      // write the EstimateSummary object
      it->Write();

      gDirectory->mkdir(it->name.c_str());
      gDirectory->cd(it->name.c_str());

      it->nominal->Write();

      histIt = it->lowHists.begin();
      histEnd = it->lowHists.end();
      for(; histIt!=histEnd; ++histIt)
        (*histIt)->Write();

      histIt = it->highHists.begin();
      histEnd = it->highHists.end();
      for(; histIt!=histEnd; ++histIt)
        (*histIt)->Write();
      
    }
  }


  TH1 * GetHisto( TFile * inFile, const string name ){

  if(!inFile || name.empty()){
    cerr << "Not all necessary info are set to access the input file. Check your config" << endl;
    cerr << "fileptr: " << inFile
         << "path/obj: " << name << endl;
    return 0;
  }
  #ifdef DEBUG
    cout << "Retrieving " << name ;
  #endif
    TH1 * ptr = (TH1 *) (inFile->Get( name.c_str() )->Clone());  
  #ifdef DEBUG
    cout << " found at " << ptr << " with integral " << ptr->Integral() << " and mean " << ptr->GetMean() << endl;
  #endif
    if (ptr) ptr->SetDirectory(0); //         for the current histogram h
    //TH1::AddDirectory(kFALSE);
    return ptr;

  }

  TH1 * GetHisto( const string file, const string path, const string obj){

  #ifdef DEBUG
    cout << "Retrieving " << file << ":" << path << obj ;
  #endif
    TFile inFile(file.c_str());
    TH1 * ptr = (TH1 *) (inFile.Get( (path+obj).c_str() )->Clone());
  #ifdef DEBUG
    cout << " found at " << ptr << " with integral " << ptr->Integral() << " and mean " << ptr->GetMean() << endl;
  #endif
    //    if(file.empty() || path.empty() || obj.empty()){
    if(!ptr){
      cerr << "Not all necessary info are set to access the input file. Check your config" << endl;
      cerr << "filename: " << file
           << "path: " << path
           << "obj: " << obj << endl;
    }
    else 
       ptr->SetDirectory(0); //         for the current histogram h

    return ptr;

  }

  void AddSubStrings( vector<string> & vs, string s){
    const string delims("\\ ");
    string::size_type begIdx, endIdx;
    begIdx=s.find_first_not_of(delims);
    while(begIdx!=string::npos){
      endIdx=s.find_first_of(delims, begIdx);
      if(endIdx==string::npos) endIdx=s.length();
      vs.push_back(s.substr(begIdx,endIdx-begIdx));
      begIdx=s.find_first_not_of(delims, endIdx);
    }
  }

  /*
  bool AddSummaries( vector<EstimateSummary> & channel, vector<vector<EstimateSummary> > &master){
    string channel_str=channel[0].channel;
    for( unsigned int proc=1;  proc < channel.size(); proc++){
      if(channel[proc].channel != channel_str){
        std::cout << "Illegal channel description, should be " << channel_str << " but found " << channel.at(proc).channel << std::endl;
        std::cout << "name " << channel.at(proc).name << std::endl;
        exit(1);
      }
      master.push_back(channel); 
    } 
    return true;
  }*/
}
}
