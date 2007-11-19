#include "TObject.h"
#include "TBigDSWriteParticle.h"

/*
Class to handle tracks witch clusters

071003 [ondrej] created, based upon Michals T[Mer,Write][Event,Particle]

*/

ClassImp(TBigDSWriteParticle)

TBigDSWriteParticle::TBigDSWriteParticle()
{
// default constructor - zeros and emty array of clusters
  fPx=0;
  fPy=0;
  fPz=0;
  fNClusters=0;
}

TBigDSWriteParticle::TBigDSWriteParticle(Float_t px,Float_t py, Float_t pz)
{
  fPx=px;
  fPy=py;
  fPz=pz;
  fNClusters=0;
}

TBigDSWriteParticle::TBigDSWriteParticle(TBigDSWriteParticle *anopart)
{
//  this->Copy(anopart);
//   Int_t nclust, fgn;
//   Float_t fftmpp[3];
//   Int_t fftmpn[4];
   Int_t nclust;
  nclust=anopart->GetNClusters();

  fNClusters=0;
  for (int i=0;i<nclust;i++)
  {
//     fgn=anopart->GetCluster(fftmpp,fftmpn,i);
//     this->AddCluster(fftmpp[0], fftmpp[1], fftmpp[2], fftmpn[0],fftmpn[1],fftmpn[2],fftmpn[3]);
  }
  fPx=anopart->GetPx();
  fPy=anopart->GetPy();
  fPz=anopart->GetPz();
  flag=anopart->GetFlag();
  fch=anopart->GetCharge();
}
  
  
TBigDSWriteParticle::~TBigDSWriteParticle()
{
// remove clusters
}

/*
Int_t TBigDSWriteParticle::AddCluster(Float_t x, Float_t y, Float_t z, Int_t pos, Int_t cha, Int_t sig, Int_t var)
{
  fClustX[fNClusters]=x;
  fClustY[fNClusters]=y;
  fClustZ[fNClusters]=z;
  fClustPos[fNClusters]=pos;
  fClustCha[fNClusters]=cha;
  fClustSig[fNClusters]=sig;
  fClustVar[fNClusters]=var;
  fNClusters++;
  return fNClusters;
}

Int_t TBigDSWriteParticle::GetCluster(Float_t clustp[3], Int_t  clustn[4], Int_t nclust)
{
  clustp[0]=fClustX[nclust];
  clustp[1]=fClustY[nclust];
  clustp[2]=fClustZ[nclust];
  clustn[0]=fClustPos[nclust];
  clustn[1]=fClustCha[nclust];
  clustn[2]=fClustSig[nclust];
  clustn[3]=fClustVar[nclust];
  return 0;
}

Int_t TBigDSWriteParticle::GetCluster(Float_t clustp[3], Int_t nclust)
{
  if(nclust < fNClusters) { 
    clustp[0]=fClustX[nclust];
    clustp[1]=fClustY[nclust];
    clustp[2]=fClustZ[nclust];
    return 0;
  } else {
    return -1;
  }
}
	
void TBigDSWriteParticle::Copy(TBigDSWriteParticle *anopart)
{
//
// Method which copy another particle to this one. During this operation is
// previous information in this particle destroyed.
//
  Int_t nclust, fgn;
  Float_t fftmpp[3];
  Int_t fftmpn[4];
  
  nclust=anopart->GetNClusters();

  fNClusters=0;
  for (int i=0;i<nclust;i++)
  {
    fgn=anopart->GetCluster(fftmpp,fftmpn,i);
    this->AddCluster(fftmpp[0], fftmpp[1], fftmpp[2], fftmpn[0],fftmpn[1],fftmpn[2],fftmpn[3]);
  }
  fPx=anopart->GetPx();
  fPy=anopart->GetPy();
  fPz=anopart->GetPz();
  flag=anopart->GetFlag();
  fch=anopart->GetCharge();
}

*/
