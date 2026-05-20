#ifndef A_HEADER
#define A_HEADER

#include <vector>
#include <iostream>
#include "TClonesArray.h"
#include "TObject.h"

class B : public TObject {
public:
    Int_t dummy;

    B() : dummy(0) {}
    B(Int_t dummyArg) : dummy(dummyArg) {}

    void SetDummy(Int_t dummyArg) { dummy = dummyArg; }

    Int_t GetDummy() const { return dummy; }

    ClassDefOverride(B, 1);
};

class A {
public:
    Int_t           num;
    std::vector<Float_t> vectorFloat;
    std::vector<B>  vectorB;
    std::vector<B*> vectorBStar;
    std::vector<B> *vectorStarB;//->
    B*              BStar;//->
    B               BArray [12];
    B               BObject;
    B*              BStarArray;//[num] // Breaks Fill()
    TClonesArray    BClonesArray;

    A()  :
        num(0),
        vectorStarB(0),
        BStar(0),
        BStarArray(0),
        BClonesArray("B")
        {  }
    A(Int_t nArg) :
        num(nArg),
        vectorStarB(new std::vector<B>()),
        BStar(new B()),
        BStarArray(new B[num]),
        BClonesArray("B")
        {  }
    virtual ~A() {  };

    void SetNum(Int_t Num) { num = Num; }
    void SetVectorStarB(std::vector<B> *vectorArg) { vectorStarB = vectorArg; }
    void SetBStar(B* bArg) { BStar = bArg; }
    //void SetBClonesArray(TClonesArray cloneArgs) { BClonesArray = cloneArgs; }

    void AddToVectorFloat(Float_t item) { vectorFloat.push_back(item); }
    void AddToVectorB(B item)       { vectorB.push_back(item); }
    void AddToVectorBStar(B *item)  { vectorBStar.push_back(item); }
    void AddToVectorStarB(B item)   { vectorStarB->push_back(item); }

    void FillBArray(B *items) { for (int i = 0; i < 12; ++i) BArray[i] = items[i]; }

    void PrintBArray() { for (int i = 0; i < 12; ++i) printf(" %i", BArray[i].GetDummy()); }

    Int_t           GetNum() const      { return num; }
    std::vector<Float_t>& GetVectorFloat() { return vectorFloat; }
    std::vector<B>& GetVectorB()        { return vectorB; }
    std::vector<B*>&GetVectorBStar()    { return vectorBStar; }
    std::vector<B>* GetVectorStarB()    { return vectorStarB; }
    B*              GetBStar()          { return BStar; }
    B*              GetBArray()         { return BArray; }
    B*              GetBStarArray()     { return BStarArray; }
    //TClonesArray  GetBClonesArray()   { return BClonesArray; }

    Float_t         GetFromVectorFloat(Int_t index) { return vectorFloat.at(index); }
    B               GetFromVectorB(Int_t index)     { return vectorB.at(index); }
    B               GetFromVectorStarB(Int_t index) { return vectorStarB->at(index); }
    B*              GetFromVectorBStar(Int_t index) { return vectorBStar.at(index); }

    void ResetVectorFloat() { vectorFloat.clear(); }
    void ResetVectorB() { vectorB.clear(); }
    void ResetVectorStarB() { vectorStarB->clear(); }
    void ResetBStarArray() { BStarArray = new B[num]; }

    ClassDef(A, 1);
};

#endif
