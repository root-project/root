//This script is to implement the vectorlized reconstruct process utilizing the matrix operation feacture that numerical librarys should have
//However, it seems that ROOT doesn't has such feature.
//It's probably better to implement this process with numpy.

#include "TMatrix.h"
#include "TRandom.h"
#include "TVectorF.h"
#include "iostream"
#include <fstream>
#include <vector>

using namespace std;

TMatrix GetRandomSignal(int nPS = 12,int nPM = 2){

    if(!gRandom)gRandom = new TRandom(0);
    TRandom& r = *gRandom;

    TMatrix signal(nPS,nPM);
    for(int i=0;i<nPS;i++){
        for(int j=0;j<nPM;j++){
            signal[i][j] = r.Uniform(0,100);
        }
    }

    return signal;

}

float GetPosRatio(float val_1,float val_2){
    return val_1/(val_1+val_2);
}

TVectorF GetPosRatioVector(TMatrix signal){
    auto raw = signal[0];
    if(signal.GetNcols()!=2){
        cout<<"Size error"<<endl;
        return TVectorF(0);
    }

    int n = signal.GetNrows();
    TVectorF ratioVector(n);

    for(int i=0;i<n;i++){
        ratioVector[i] = GetPosRatio(signal[i][0],signal[i][1]);
    }

    return ratioVector;
}

//
TMatrix GetPMTMatFromTxt(string path){

    fstream file(path);
    if(!file.is_open()){
        cout<<"File not found"<<endl;
        return TMatrix(0,0);
    }

    vector<vector<float>> mat;
    while(1){
        float x,y,z;
        file>>x>>y>>z;
        if(file.eof()){
            break;
        }
        else{
            auto v = vector<float>{x,y,z};
            mat.push_back(v);
        }
    }

    TMatrix tMat(mat.size(),mat[0].size());
    for(int i=0;i<mat.size();i++)
        for(int j=0;j<mat[0].size();j++)
            tMat[i][j] = mat[i][j];

    return tMat;
}

