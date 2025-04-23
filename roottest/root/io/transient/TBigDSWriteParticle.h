#ifndef TBigDSWriteParticle_Header

#define TBigDSWriteParticle_Header

#include "TObject.h"

#define fgMaxClust 200                  // max clusters / track

class TBigDSWriteParticle : public TObject
{
  private :
    Float_t fPx;                        // Momenta in x direction in LAB
    Float_t fPy;                        // Momenta in y direction in LAB
    Float_t fPz;                        // Momenta in z direction in LAB
    Int_t fNClusters;                   // Number of clusters in particle track
/*    Float_t fClustX[fgMaxClust];
    Float_t fClustY[fgMaxClust];
    Float_t fClustZ[fgMaxClust];
    Int_t fClustPos[fgMaxClust];
    Int_t fClustCha[fgMaxClust];
    Int_t fClustSig[fgMaxClust];
    Int_t fClustVar[fgMaxClust];*/
    Short_t flag;                       // inflag
    Short_t fch;                        // charge
   
  public :
    TBigDSWriteParticle();
    TBigDSWriteParticle(TBigDSWriteParticle *anopart);
    TBigDSWriteParticle(Float_t px,Float_t py, Float_t pz);
    ~TBigDSWriteParticle();

    void SetPx(Float_t px) {fPx=px;};                   // Set px
    void SetPy(Float_t py) {fPy=py;};                   // Set py
    void SetPz(Float_t pz) {fPz=pz;};                   // Set pz
    Float_t GetPx() {return fPx;};                      // Returns px
    Float_t GetPy() {return fPy;};                      // Returns py
    Float_t GetPz() {return fPz;};                      // Returns pz
    void SetCharge(Short_t ch) {fch=ch;};               // Set charge
    void SetFlag(Short_t ff) {flag=ff;};                // Set flag
    Short_t GetCharge() {return fch;};                  // Get particle charge
    Short_t GetCh() {return fch;};                      // Get particle charge
    Short_t GetFlag() {return flag;};                   // Get flag
    
    Int_t GetNClusters()  {return fNClusters;};         // Get number of clusters
    void SetNClusters(Int_t clusters) {fNClusters=clusters;};
/*    Int_t AddCluster(Float_t x, Float_t y, Float_t z, Int_t pos, Int_t cha, Int_t sig, Int_t var); // Add cluster
    Int_t GetCluster(Float_t clustp[3], Int_t  clustn[4], Int_t nclust);                // Get cluster
    Int_t GetCluster(Float_t clustp[3], Int_t nclust);          // Get cluster
    //void Copy(TBigDSWriteParticle *anopart);                  // Copy another particle to this particle
*/      
  ClassDef(TBigDSWriteParticle,1) // Event base class
 
};

#endif
