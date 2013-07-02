#include <vector>
#include <string>
#include "TH1.h"
#include "TTree.h"
#include "TFile.h"
#include "TFolder.h"

class CustomInnerObject {
public:
  CustomObject(): fData("inner") {}
  CustomObject(const char* data): fData(data) {}
  std::string fData;
};
class CustomBaseObject {
public:
  CustomBaseObject(): fData("base") {}
  CustomBaseObject(const char* data): fData(data) {}
  std::string fData;
};

class CustomOuterObject:
  public CustomBaseObject,
  public std::vector<CustomBaseObject> {
public:
CustomObject(): fOuterData("outer") {}
  CustomObject(const char* data, const char* innerdata, const char* basedata):
    CustomBaseObject(basedata), fOuterData(data), fInnerData(innerdata) {}
  std::string fOuterData;
  CustomInnerObject fInnerData;
};

void makeFeatureTree() {
  TFile* file = TFile::Open("featuretree.root", "RECREATE");
  TTree* tree = new TTree("T", "Feature Tree");

  TList objects99;
  objects99.AddLast(new TNamed("first99", "First Object"));
  TH1F* hist99 = new TH1F("hSecond99", "second, a histogram", 100, 0., 1.);
  objects99.AddLast(hist99);
  tree->Branch(&objects99);

  TList objects1;
  objects1.AddLast(new TNamed("first1", "First Object"));
  TH1F* hist1 = new TH1F("hSecond1", "second, a histogram", 100, 0., 1.);
  objects1.AddLast(hist1);
  tree->Branch(&objects1, 32000, 1);

  TList objects0;
  objects0.AddLast(new TNamed("first0", "First Object"));
  TH1F* hist0 = new TH1F("hSecond0", "second, a histogram", 100, 0., 1.);
  objects0.AddLast(hist0);
  tree->Branch(&objects0, 32000, 0);

  TFolder folder99("Folder99");
  folder99.AddLast(new TNamed("firstf99", "First Object"));
  TH1F* histf99 = new TH1F("hSecondFoler99", "second, a histogram", 100, 0., 1.);
  folder99.AddLast(histf99);
  tree->Branch("Folder99");
  
  TFolder folder1("Folder1");
  folder1.AddLast(new TNamed("firstf1", "First Object"));
  TH1F* histf1 = new TH1F("hSecondFoler1", "second, a histogram", 100, 0., 1.);
  folder1.AddLast(histf1);
  tree->Branch("Folder1", 32000, 1);

  TFolder folder0("Folder0");
  folder0.AddLast(new TNamed("firstf0", "First Object"));
  TH1F* histf0 = new TH1F("hSecondFoler0", "second, a histogram", 100, 0., 1.);
  folder0.AddLast(histf0);
  tree->Branch("Folder0", 32000, 0);
  

  TH1F* 
}