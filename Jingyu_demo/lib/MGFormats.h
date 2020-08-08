#include "TFile.h"
#include <iostream>
#include <fstream>
#include "TTree.h"
#include "TVector3.h"
#include "TObjArray.h"
#include "TList.h"
#include "TArray.h"

using namespace std;

void ConvertLogToRoot(string fileName){

    fstream inputFile(fileName);
    if(!inputFile.is_open()){
        cout<<"File not found"<<endl;
        return;
    }

    TList posList;

    while (1)
    {
        float x,y,z;
        inputFile>>x>>y>>z;
        if(inputFile.eof())break;
        else{
            TVector3* pos = new TVector3(x,y,z);
            posList.Add(pos);
        }
    }

    TFile outpuFile("reconstruction/data/reconstructedVertex.root","RECREATE");

    posList.Write();

    outpuFile.Close();

}

void ConvertPMTPosToRoot(string fileName){
    fstream inputFile(fileName);
    if(!inputFile.is_open()){
        cout<<"File not found"<<endl;
        return;
    }

    TList posList;
}