// @(#)root/tmva $Id$
// Author: Rustem Ospanov 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : ModulekNN                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Author:                                                                        *
 *      Rustem Ospanov <rustem@fnal.gov> - U. of Texas at Austin, USA             *
 *                                                                                *
 * Copyright (c) 2007:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      U. of Texas at Austin, USA                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/ModulekNN.h"

// C++
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "TMath.h"

// TMVA
#include "TMVA/MsgLogger.h"
 
//-------------------------------------------------------------------------------------------
TMVA::kNN::Event::Event() 
   :fVar(),
    fWeight(-1.0),
    fType(-1)
{
   // default constructor
}

//-------------------------------------------------------------------------------------------
TMVA::kNN::Event::Event(const VarVec &var, const Double_t weight, const Short_t type)
   :fVar(var),
    fWeight(weight),
    fType(type)
{
   // constructor
}

//-------------------------------------------------------------------------------------------
TMVA::kNN::Event::Event(const VarVec &var, const Double_t weight, const Short_t type, const VarVec &tvec)
   :fVar(var),
    fTgt(tvec),
    fWeight(weight),
    fType(type)
{
   // constructor
}

//-------------------------------------------------------------------------------------------
TMVA::kNN::Event::~Event()
{
   // destructor
}

//-------------------------------------------------------------------------------------------
TMVA::kNN::VarType TMVA::kNN::Event::GetDist(const Event &other) const
{
   // compute distance
   const UInt_t nvar = GetNVar();

   if (nvar != other.GetNVar()) {
      std::cerr << "Distance: two events have different dimensions" << std::endl;
      return -1.0;
   }
   
   VarType sum = 0.0;
   for (UInt_t ivar = 0; ivar < nvar; ++ivar) {
      sum += GetDist(other.GetVar(ivar), ivar);
   }
   
   return sum;
}

//-------------------------------------------------------------------------------------------
void TMVA::kNN::Event::SetTargets(const VarVec &tvec)
{
   fTgt = tvec;
}

//-------------------------------------------------------------------------------------------
const TMVA::kNN::VarVec& TMVA::kNN::Event::GetTargets() const
{
   return fTgt;
}

//-------------------------------------------------------------------------------------------
const TMVA::kNN::VarVec& TMVA::kNN::Event::GetVars() const
{
   return fVar;
}

//-------------------------------------------------------------------------------------------
void TMVA::kNN::Event::Print() const
{
   // print
   Print(std::cout);
}

//-------------------------------------------------------------------------------------------
void TMVA::kNN::Event::Print(std::ostream& os) const
{
   // print
   Int_t dp = os.precision();
   os << "Event: ";
   for (UInt_t ivar = 0; ivar != GetNVar(); ++ivar) {
      if (ivar == 0) {
         os << "(";
      }
      else {
         os << ", ";
      }

      os << std::setfill(' ') << std::setw(5) << std::setprecision(3) << GetVar(ivar);
   }

   if (GetNVar() > 0) {
      os << ")";
   }
   else {
      os << " no variables";
   }
   os << std::setprecision(dp);
}

//-------------------------------------------------------------------------------------------
std::ostream& TMVA::kNN::operator<<(std::ostream& os, const TMVA::kNN::Event& event)
{
   // streamer
   event.Print(os);
   return os;
}





TRandom3 TMVA::kNN::ModulekNN::fgRndm(1);

//-------------------------------------------------------------------------------------------
TMVA::kNN::ModulekNN::ModulekNN()
   :fDimn(0),
    fTree(0),
    fLogger( new MsgLogger("ModulekNN") )
{
   // default constructor
}

//-------------------------------------------------------------------------------------------
TMVA::kNN::ModulekNN::~ModulekNN()
{
   // destructor
   if (fTree) {
      delete fTree; fTree = 0;
   }
   delete fLogger;
}

//-------------------------------------------------------------------------------------------
void TMVA::kNN::ModulekNN::Clear()
{
   // clean up
   fDimn = 0;

   if (fTree) {
      delete fTree;
      fTree = 0;
   }

   fVarScale.clear();
   fCount.clear();
   fEvent.clear();
   fVar.clear();
}

//-------------------------------------------------------------------------------------------
void TMVA::kNN::ModulekNN::Add(const Event &event)
{   
   // add an event to tree
   if (fTree) {
      Log() << kFATAL << "<Add> Cannot add event: tree is already built" << Endl;
      return;      
   }

   if (fDimn < 1) {
      fDimn = event.GetNVar();
   }
   else if (fDimn != event.GetNVar()) {
      Log() << kFATAL << "ModulekNN::Add() - number of dimension does not match previous events" << Endl;
      return;
   }

   fEvent.push_back(event);

   for (UInt_t ivar = 0; ivar < fDimn; ++ivar) {
      fVar[ivar].push_back(event.GetVar(ivar));
   }

   std::map<Short_t, UInt_t>::iterator cit = fCount.find(event.GetType());
   if (cit == fCount.end()) {
      fCount[event.GetType()] = 1;
   }
   else {
      ++(cit->second);
   }
}

//-------------------------------------------------------------------------------------------
Bool_t TMVA::kNN::ModulekNN::Fill(const UShort_t odepth, const UInt_t ifrac, const std::string &option)
{
   // fill the tree
   if (fTree) {
      Log() << kFATAL << "ModulekNN::Fill - tree has already been created" << Endl;
      return kFALSE;
   }   

   // If trim option is set then find class with lowest number of events
   // and set that as maximum number of events for all other classes.
   UInt_t min = 0;
   if (option.find("trim") != std::string::npos) {
      for (std::map<Short_t, UInt_t>::const_iterator it = fCount.begin(); it != fCount.end(); ++it) {
         if (min == 0 || min > it->second) {
            min = it->second;
         }
      }
      
      Log() << kINFO << "<Fill> Will trim all event types to " << min << " events" << Endl;
      
      fCount.clear();
      fVar.clear();
      
      EventVec evec;
      
      for (EventVec::const_iterator event = fEvent.begin(); event != fEvent.end(); ++event) {
         std::map<Short_t, UInt_t>::iterator cit = fCount.find(event->GetType());
         if (cit == fCount.end()) {
            fCount[event->GetType()] = 1;
         }
         else if (cit->second < min) {
            ++(cit->second);
         }
         else {
            continue;
         }

         for (UInt_t d = 0; d < fDimn; ++d) {
            fVar[d].push_back(event->GetVar(d));
         }

         evec.push_back(*event);
      }

      Log() << kINFO << "<Fill> Erased " << fEvent.size() - evec.size() << " events" << Endl;
      
      fEvent = evec;
   }

   // clear event count
   fCount.clear();

   // sort each variable for all events - needs this before Optimize() and ComputeMetric()
   for (VarMap::iterator it = fVar.begin(); it != fVar.end(); ++it) {
      std::sort((it->second).begin(), (it->second).end());
   }

   if (option.find("metric") != std::string::npos && ifrac > 0) {
      ComputeMetric(ifrac);
      
      // sort again each variable for all events - needs this before Optimize()
      // rescaling has changed variable values
      for (VarMap::iterator it = fVar.begin(); it != fVar.end(); ++it) {
         std::sort((it->second).begin(), (it->second).end());
      }
   }

   // If odepth > 0 then fill first odepth levels
   // with empty nodes that split separating variable in half for
   // all child nodes. If odepth = 0 then split variable 0
   // at the median (in half) and return it as root node
   fTree = Optimize(odepth);
   
   if (!fTree) {
      Log() << kFATAL << "ModulekNN::Fill() - failed to create tree" << Endl;
      return kFALSE;      
   }      
   
   for (EventVec::const_iterator event = fEvent.begin(); event != fEvent.end(); ++event) {
      fTree->Add(*event, 0);
      
      std::map<Short_t, UInt_t>::iterator cit = fCount.find(event->GetType());
      if (cit == fCount.end()) {
         fCount[event->GetType()] = 1;
      }
      else {
         ++(cit->second);
      }
   }
   
   for (std::map<Short_t, UInt_t>::const_iterator it = fCount.begin(); it != fCount.end(); ++it) {
      Log() << kINFO << "<Fill> Class " << it->first << " has " << std::setw(8) 
              << it->second << " events" << Endl;
   }
   
   return kTRUE;
}

//-------------------------------------------------------------------------------------------
Bool_t TMVA::kNN::ModulekNN::Find(Event event, const UInt_t nfind, const std::string &option) const
{  
   // find in tree
   // if tree has been filled then search for nfind closest events 
   // if metic (fVarScale map) is computed then rescale event variables
   // using previsouly computed width of variable distribution

   if (!fTree) {
      Log() << kFATAL << "ModulekNN::Find() - tree has not been filled" << Endl;
      return kFALSE;
   }
   if (fDimn != event.GetNVar()) {
      Log() << kFATAL << "ModulekNN::Find() - number of dimension does not match training events" << Endl;
      return kFALSE;
   }
   if (nfind < 1) {
      Log() << kFATAL << "ModulekNN::Find() - requested 0 nearest neighbors" << Endl;
      return kFALSE;
   }

   // if variable widths are computed then rescale variable in this event
   // to same widths as events in stored kd-tree
   if (!fVarScale.empty()) {
      event = Scale(event);
   }

   // latest event for k-nearest neighbor search
   fkNNEvent = event;
   fkNNList.clear();
   
   if(option.find("weight") != std::string::npos)
   {
      // recursive kd-tree search for nfind-nearest neighbors
      // use event weight to find all nearest events
      // that have sum of weights >= nfind
      kNN::Find<kNN::Event>(fkNNList, fTree, event, Double_t(nfind), 0.0);
   }
   else
   {
      // recursive kd-tree search for nfind-nearest neighbors
      // count nodes and do not use event weight
      kNN::Find<kNN::Event>(fkNNList, fTree, event, nfind);      
   }

   return kTRUE;
}

//-------------------------------------------------------------------------------------------
Bool_t TMVA::kNN::ModulekNN::Find(const UInt_t nfind, const std::string &option) const
{
   // find in tree
   if (fCount.empty() || !fTree) {
      return kFALSE;
   }
   
   static std::map<Short_t, UInt_t>::const_iterator cit = fCount.end();

   if (cit == fCount.end()) {
      cit = fCount.begin();
   }

   const Short_t etype = (cit++)->first;

   if (option == "flat") {
      VarVec dvec;
      for (UInt_t d = 0; d < fDimn; ++d) {
         VarMap::const_iterator vit = fVar.find(d);
         if (vit == fVar.end()) {
            return kFALSE;
         }
         
         const std::vector<Double_t> &vvec = vit->second;
         
         if (vvec.empty()) {
            return kFALSE;
         }

         // assume that vector elements of fVar are sorted
         const VarType min = vvec.front();
         const VarType max = vvec.back();
         const VarType width = max - min;
         
         if (width < 0.0 || width > 0.0) {
            dvec.push_back(min + width*fgRndm.Rndm());
         }
         else {
            return kFALSE;
         }
      }

      const Event event(dvec, 1.0, etype);
      
      Find(event, nfind);
      
      return kTRUE;
   }

   return kFALSE;
}

//-------------------------------------------------------------------------------------------
TMVA::kNN::Node<TMVA::kNN::Event>* TMVA::kNN::ModulekNN::Optimize(const UInt_t odepth)
{
   // Optimize() balances binary tree for first odepth levels
   // for each depth we split sorted depth % dimension variables
   // into 2^odepth parts

   if (fVar.empty() || fDimn != fVar.size()) {
      Log() << kWARNING << "<Optimize> Cannot build a tree" << Endl;
      return 0;
   }

   const UInt_t size = (fVar.begin()->second).size();
   if (size < 1) {
      Log() << kWARNING << "<Optimize> Cannot build a tree without events" << Endl;
      return 0;
   }

   VarMap::const_iterator it = fVar.begin();
   for (; it != fVar.end(); ++it) {
      if ((it->second).size() != size) {
         Log() << kWARNING << "<Optimize> # of variables doesn't match between dimensions" << Endl;
         return 0;
      }
   }

   if (double(fDimn*size) < TMath::Power(2.0, double(odepth))) {
      Log() << kWARNING << "<Optimize> Optimization depth exceeds number of events" << Endl;
      return 0;      
   }   

   Log() << kINFO << "Optimizing tree for " << fDimn << " variables with " << size << " values" << Endl;

   std::vector<Node<Event> *> pvec, cvec;

   it = fVar.find(0);
   if (it == fVar.end() || (it->second).size() < 2) {
      Log() << kWARNING << "<Optimize> Missing 0 variable" << Endl;
      return 0;
   }

   const Event pevent(VarVec(fDimn, (it->second)[size/2]), -1.0, -1);
   
   Node<Event> *tree = new Node<Event>(0, pevent, 0);
   
   pvec.push_back(tree);

   for (UInt_t depth = 1; depth < odepth; ++depth) {            
      const UInt_t mod = depth % fDimn;
      
      VarMap::const_iterator vit = fVar.find(mod);
      if (vit == fVar.end()) {
         Log() << kFATAL << "Missing " << mod << " variable" << Endl;
         return 0;
      }
      const std::vector<Double_t> &dvec = vit->second;

      if (dvec.size() < 2) {
         Log() << kFATAL << "Missing " << mod << " variable" << Endl;
         return 0;
      }
      
      UInt_t ichild = 1;
      for (std::vector<Node<Event> *>::iterator pit = pvec.begin(); pit != pvec.end(); ++pit) {
         Node<Event> *parent = *pit;
         
         const VarType lmedian = dvec[size*ichild/(2*pvec.size() + 1)];
         ++ichild;

         const VarType rmedian = dvec[size*ichild/(2*pvec.size() + 1)];
         ++ichild;
      
         const Event levent(VarVec(fDimn, lmedian), -1.0, -1);
         const Event revent(VarVec(fDimn, rmedian), -1.0, -1);
         
         Node<Event> *lchild = new Node<Event>(parent, levent, mod);
         Node<Event> *rchild = new Node<Event>(parent, revent, mod);
         
         parent->SetNodeL(lchild);
         parent->SetNodeR(rchild);
         
         cvec.push_back(lchild);
         cvec.push_back(rchild);
      }

      pvec = cvec;
      cvec.clear();
   }
   
   return tree;
}

//-------------------------------------------------------------------------------------------
void TMVA::kNN::ModulekNN::ComputeMetric(const UInt_t ifrac)
{
   // compute scale factor for each variable (dimension) so that 
   // distance is computed uniformely along each dimension
   // compute width of interval that includes (100 - 2*ifrac)% of events
   // below, assume that in fVar each vector of values is sorted

   if (ifrac == 0) {
      return;
   }
   if (ifrac > 100) {
      Log() << kFATAL << "ModulekNN::ComputeMetric - fraction can not exceed 100%" << Endl;
      return;
   }   
   if (!fVarScale.empty()) {
      Log() << kFATAL << "ModulekNN::ComputeMetric - metric is already computed" << Endl;
      return;
   }
   if (fEvent.size() < 100) {
      Log() << kFATAL << "ModulekNN::ComputeMetric - number of events is too small" << Endl;
      return;      
   }

   const UInt_t lfrac = (100 - ifrac)/2;
   const UInt_t rfrac = 100 - (100 - ifrac)/2;

   Log() << kINFO << "Computing scale factor for 1d distributions: " 
           << "(ifrac, bottom, top) = (" << ifrac << "%, " << lfrac << "%, " << rfrac << "%)" << Endl;   

   fVarScale.clear();
   
   for (VarMap::const_iterator vit = fVar.begin(); vit != fVar.end(); ++vit) {
      const std::vector<Double_t> &dvec = vit->second;
      
      std::vector<Double_t>::const_iterator beg_it = dvec.end();
      std::vector<Double_t>::const_iterator end_it = dvec.end();
      
      Int_t dist = 0;
      for (std::vector<Double_t>::const_iterator dit = dvec.begin(); dit != dvec.end(); ++dit, ++dist) {
         
         if ((100*dist)/dvec.size() == lfrac && beg_it == dvec.end()) {
            beg_it = dit;
         }
         
         if ((100*dist)/dvec.size() == rfrac && end_it == dvec.end()) {
            end_it = dit;
         }
      }

      if (beg_it == dvec.end() || end_it == dvec.end()) {
         beg_it = dvec.begin();
         end_it = dvec.end();
         
         assert(beg_it != end_it && "Empty vector");
         
         --end_it;
      }

      const Double_t lpos = *beg_it;
      const Double_t rpos = *end_it;
      
      if (!(lpos < rpos)) {
         Log() << kFATAL << "ModulekNN::ComputeMetric() - min value is greater than max value" << Endl;
         continue;
      }
      
      // Rustem: please find a solution that does not use distance (it does not exist on solaris)
      //       Log() << kINFO << "Variable " << vit->first 
      //               << " included " << distance(beg_it, end_it) + 1
      //               << " events: width = " << std::setfill(' ') << std::setw(5) << std::setprecision(3) << rpos - lpos
      //               << ", (min, max) = (" << std::setfill(' ') << std::setw(5) << std::setprecision(3) << lpos 
      //               << ", " << std::setfill(' ') << std::setw(5) << std::setprecision(3) << rpos << ")" << Endl;
      
      fVarScale[vit->first] = rpos - lpos;
   }

   fVar.clear();

   for (UInt_t ievent = 0; ievent < fEvent.size(); ++ievent) {      
      fEvent[ievent] = Scale(fEvent[ievent]);
      
      for (UInt_t ivar = 0; ivar < fDimn; ++ivar) {
         fVar[ivar].push_back(fEvent[ievent].GetVar(ivar));
      }
   }
}

//-------------------------------------------------------------------------------------------
const TMVA::kNN::Event TMVA::kNN::ModulekNN::Scale(const Event &event) const
{
   // scale each event variable so that rms of variables is approximately 1.0
   // this allows comparisons of variables with distinct scales and units
   
   if (fVarScale.empty()) {
      return event;
   }

   if (event.GetNVar() != fVarScale.size()) {
      Log() << kFATAL << "ModulekNN::Scale() - mismatched metric and event size" << Endl;
      return event;
   }

   VarVec vvec(event.GetNVar(), 0.0);

   for (UInt_t ivar = 0; ivar < event.GetNVar(); ++ivar) {
      std::map<int, Double_t>::const_iterator fit = fVarScale.find(ivar);
      if (fit == fVarScale.end()) {
         Log() << kFATAL << "ModulekNN::Scale() - failed to find scale for " << ivar << Endl;
         continue;
      }
      
      if (fit->second > 0.0) {
         vvec[ivar] = event.GetVar(ivar)/fit->second;
      }
      else {
         Log() << kFATAL << "Variable " << ivar << " has zero width" << Endl;
      }
   }

   return Event(vvec, event.GetWeight(), event.GetType(), event.GetTargets());
}

//-------------------------------------------------------------------------------------------
void TMVA::kNN::ModulekNN::Print() const
{
   // print
   Print(std::cout);
}

//-------------------------------------------------------------------------------------------
void TMVA::kNN::ModulekNN::Print(ostream &os) const
{
   // print
   os << "----------------------------------------------------------------------"<< std::endl;
   os << "Printing knn result" << std::endl;
   os << fkNNEvent << std::endl;

   UInt_t count = 0;

   std::map<Short_t, Double_t> min, max;

   os << "Printing " << fkNNList.size() << " nearest neighbors" << std::endl;
   for (List::const_iterator it = fkNNList.begin(); it != fkNNList.end(); ++it) {
      os << ++count << ": " << it->second << ": " << it->first->GetEvent() << std::endl;
      
      const Event &event = it->first->GetEvent();
      for (UShort_t ivar = 0; ivar < event.GetNVar(); ++ivar) {
         if (min.find(ivar) == min.end()) {
            min[ivar] = event.GetVar(ivar);
         }
         else if (min[ivar] > event.GetVar(ivar)) {
            min[ivar] = event.GetVar(ivar);
         }

         if (max.find(ivar) == max.end()) {
            max[ivar] = event.GetVar(ivar);
         }
         else if (max[ivar] < event.GetVar(ivar)) {
            max[ivar] = event.GetVar(ivar);
         }
      }
   }

   if (min.size() == max.size()) {
      for (std::map<Short_t, Double_t>::const_iterator mit = min.begin(); mit != min.end(); ++mit) {
         const Short_t i = mit->first;
         Log() << kINFO << "(var, min, max) = (" << i << "," << min[i] << ", " << max[i] << ")" << Endl;
      }
   }
   
   os << "----------------------------------------------------------------------" << std::endl;
}
