/// \file ROOT/RFieldVisitor.hxx
/// \ingroup NTuple ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-06-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#include "ROOT/RNTuple.hxx"
#include "ROOT/RFieldVisitor.hxx"
#include "ROOT/RField.hxx"
#include <iostream>
#include <iomanip>


void ROOT::Experimental::RPrintVisitor::visitField(const ROOT::Experimental::Detail::RFieldBase& fField) { std::cout << "Called visitField for any normal field\n";}
void ROOT::Experimental::RPrintVisitor::visitField(const ROOT::Experimental::RFieldRoot& fRootField) {
    std::cout << "Called function for RootField\n";
}
/*void ROOT::Experimental::RPrintVisitor::visitField(ROOT::Experimental::Detail::RFieldBase* fField, int fIndex) {
    /// content here: Each line has 69 char.
    //std::cout << "Calling visitField\n";
    std::cout << "* Br " << std::setw(5-NumDigits(fIndex)) << fIndex << " : " << fField->GetName() << ": " << fField->GetType() << std::setw(56-(fField->GetName().size()) - (fField->GetType().size())) <<"*\n";
    std::cout << "* Entries : \n"; //Does this even make sense?!
    std::cout << "* Baskets : \n"; //WTF is a basket.
    std::cout << "*********************************************************************\n";
}

void ROOT::Experimental::RPrintVisitor::visitField(ROOT::Experimental::RField<std::string, void>* fField, int fIndex) {
    std::cout << "* Br " << std::setw(5-NumDigits(fIndex)) << fIndex << " : " << fField->GetName() << ": " << fField->GetType() << std::setw(56-(fField->GetName().size()) - (fField->GetType().size())) <<"*\n";
    std::cout << "* Invoked templated version because data type was std::string" << std::setw(9) << "*\n";
    std::cout << "*********************************************************************\n";
}*/

int ROOT::Experimental::NumDigits(int x)
{
    x = abs(x);
    return (x < 10 ? 1 :
            (x < 100 ? 2 :
             (x < 1000 ? 3 :
              (x < 10000 ? 4 :
               (x < 100000 ? 5 :
                (x < 1000000 ? 6 :
                 (x < 10000000 ? 7 :
                  (x < 100000000 ? 8 :
                   (x < 1000000000 ? 9 :
                    10)))))))));
}
    /*
    std::cout << "****************************** NTUPLE *******************************\n";
    std::cout << "* Ntuple  : " << fField->GetName() << std::setw(58-fField->GetName().size()) << "*\n";
    std::cout << "*********************************************************************\n";*/




/*
void ROOT::Experimental::TNtuplePrintVisitor::visitNtuple(ROOT::Experimental::RNTupleReader* fReader)  {
    /// content here: Each line has 69 char.
    auto Rootfield = fReader->GetModel()->GetRootField();
    std::cout << "****************************** NTUPLE *******************************\n";
    std::cout << "* Ntuple  : " << fReader->getName() << std::setw(58-fReader->getName().size()) << "*\n";
    std::cout << "* Entries : " << fReader->GetNEntries() << std::setw(58-NumDigits(fReader->GetNEntries())) << "*\n";
    std::cout << "*********************************************************************\n";
    /// From here on entry specific for each branch.
    for(size_t i = 0; i < Rootfield->fSubFields.size(); ++i) {
    // for(size_t i = 0; i < fReader->GetModel()->GetRootField()->(Detail::GetNItems()); ++i) {
        std::cout << "* Br " << std::setw(5-NumDigits(i)) << i << " : " << Rootfield->fSubFields.at(i)->fName << ": " << Rootfield->fSubFields.at(i)->fType << std::setw(56-(Rootfield->fSubFields.at(i)->fName.size()) - (Rootfield->fSubFields.at(i)->fType.size())) <<"*\n";
        std::cout << "* Entries : \n"; //Does this even make sense?!
        std::cout << "* Baskets : \n"; //WTF is a basket.
        std::cout << "*********************************************************************\n";
    } // end for loop
    // for(auto subfieldptr: fReader->GetModel()->GetRootField()->fSubFields) {  }
    
    
    
}
*/
