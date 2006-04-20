// @(#)root/vmc:$Name:  $:$Id: TVirtualMCGeometry.h,v 1.6 2005/11/18 21:20:15 brun Exp $
// Authors: ... 25/06/2002

#ifndef ROOT_TVirtualMCGeometry
#define ROOT_TVirtualMCGeometry

//
// Class TVirtualMCGeometry
// -------------------------
// Interface to Monte Carlo geometry construction
// (separated from VirtualMC)

#include "TNamed.h"

class TGeoHMatrix;
class TArrayD;
class TString;

class TVirtualMCGeometry : public TNamed {

  public:
    TVirtualMCGeometry(const char *name, const char *title);
    TVirtualMCGeometry();
    virtual ~TVirtualMCGeometry();

    // static access method
    static TVirtualMCGeometry* Instance() { return fgInstance; }

    // detector composition
    virtual void  Material(Int_t& kmat, const char* name, Double_t a,
                     Double_t z, Double_t dens, Double_t radl, Double_t absl,
                     Float_t* buf, Int_t nwbuf) = 0;
    virtual void  Material(Int_t& kmat, const char* name, Double_t a,
                     Double_t z, Double_t dens, Double_t radl, Double_t absl,
                     Double_t* buf, Int_t nwbuf) = 0;
    virtual void  Mixture(Int_t& kmat, const char *name, Float_t *a,
                     Float_t *z, Double_t dens, Int_t nlmat, Float_t *wmat) = 0;
    virtual void  Mixture(Int_t& kmat, const char *name, Double_t *a,
                     Double_t *z, Double_t dens, Int_t nlmat, Double_t *wmat) = 0;
    virtual void  Medium(Int_t& kmed, const char *name, Int_t nmat,
                     Int_t isvol, Int_t ifield, Double_t fieldm, Double_t tmaxfd,
                     Double_t stemax, Double_t deemax, Double_t epsil,
                     Double_t stmin, Float_t* ubuf, Int_t nbuf) = 0;
    virtual void  Medium(Int_t& kmed, const char *name, Int_t nmat,
                     Int_t isvol, Int_t ifield, Double_t fieldm, Double_t tmaxfd,
                     Double_t stemax, Double_t deemax, Double_t epsil,
                     Double_t stmin, Double_t* ubuf, Int_t nbuf) = 0;
    virtual void  Matrix(Int_t& krot, Double_t thetaX, Double_t phiX,
                     Double_t thetaY, Double_t phiY, Double_t thetaZ,
                     Double_t phiZ) = 0;

    // functions from GGEOM
    virtual Int_t  Gsvolu(const char *name, const char *shape, Int_t nmed,
                          Float_t *upar, Int_t np) = 0;
    virtual Int_t  Gsvolu(const char *name, const char *shape, Int_t nmed,
                          Double_t *upar, Int_t np) = 0;
    virtual void  Gsdvn(const char *name, const char *mother, Int_t ndiv,
                         Int_t iaxis) = 0;
    virtual void  Gsdvn2(const char *name, const char *mother, Int_t ndiv,
                         Int_t iaxis, Double_t c0i, Int_t numed) = 0;
    virtual void  Gsdvt(const char *name, const char *mother, Double_t step,
                         Int_t iaxis, Int_t numed, Int_t ndvmx) = 0;
    virtual void  Gsdvt2(const char *name, const char *mother, Double_t step,
                         Int_t iaxis, Double_t c0, Int_t numed, Int_t ndvmx) = 0;
    virtual void  Gsord(const char *name, Int_t iax) = 0;
    virtual void  Gspos(const char *name, Int_t nr, const char *mother,
                         Double_t x, Double_t y, Double_t z, Int_t irot,
                         const char *konly="ONLY") = 0;
    virtual void  Gsposp(const char *name, Int_t nr, const char *mother,
                         Double_t x, Double_t y, Double_t z, Int_t irot,
                         const char *konly, Float_t *upar, Int_t np) = 0;
    virtual void  Gsposp(const char *name, Int_t nr, const char *mother,
                         Double_t x, Double_t y, Double_t z, Int_t irot,
                         const char *konly, Double_t *upar, Int_t np) = 0;
    virtual void  Gsbool(const char* onlyVolName, const char* manyVolName) = 0;



    // functions for access to geometry
    //
    // Return the Transformation matrix between the volume specified by
    // the path volumePath and the top or master volume.
    virtual Bool_t GetTransformation(const TString& volumePath,
                         TGeoHMatrix& matrix) = 0;

    // Return the name of the shape and its parameters for the volume
    // specified by the volume name.
    virtual Bool_t GetShape(const TString& volumePath,
                         TString& shapeType, TArrayD& par) = 0;

    // Returns the material parameters for the volume specified by
    // the volume name.
    virtual Bool_t GetMaterial(const TString& volumeName,
                               TString& name, Int_t& imat,
                               Double_t& a, Double_t& z, Double_t& density,
                               Double_t& radl, Double_t& inter, TArrayD& par) = 0;

    // Returns the medium parameters for the volume specified by the
    // volume name.
    virtual Bool_t GetMedium(const TString& volumeName,
                             TString& name, Int_t& imed,
                             Int_t& nmat, Int_t& isvol, Int_t& ifield,
                             Double_t& fieldm, Double_t& tmaxfd, Double_t& stemax,
                             Double_t& deemax, Double_t& epsil, Double_t& stmin,
                             TArrayD& par) = 0;

    // functions for drawing
    //virtual void  DrawOneSpec(const char* name) = 0;
    //virtual void  Gsatt(const char* name, const char* att, Int_t val) = 0;
    //virtual void  Gdraw(const char*,Double_t theta = 30, Double_t phi = 30,
    //                    Double_t psi = 0, Double_t u0 = 10, Double_t v0 = 10,
    //                    Double_t ul = 0.01, Double_t vl = 0.01) = 0;

    // Euclid
    // virtual void  WriteEuclid(const char*, const char*, Int_t, Int_t) = 0;

    // get methods
    virtual Int_t VolId(const Text_t* volName) const = 0;
    virtual const char* VolName(Int_t id) const = 0;
    virtual Int_t NofVolumes() const = 0;
    virtual Int_t NofVolDaughters(const char* volName) const;
    virtual const char*  VolDaughterName(const char* volName, Int_t i) const;
    virtual Int_t        VolDaughterCopyNo(const char* volName, Int_t i) const;
            // New functions
            // Make them = 0 with the next release
    virtual Int_t VolId2Mate(Int_t id) const = 0;

  protected:
    TVirtualMCGeometry(const TVirtualMCGeometry &mc) : TNamed(mc) {}
    TVirtualMCGeometry & operator=(const TVirtualMCGeometry &) {return (*this);}

    static TVirtualMCGeometry*  fgInstance; // singleton instance

  ClassDef(TVirtualMCGeometry,1)  //Interface to Monte Carlo geometry construction
};

// inline fuctions

// Temporary implementation of new functions
// To be removed with the next release

inline Int_t TVirtualMCGeometry::NofVolDaughters(const char* /*volName*/) const {
  Warning("NofVolDaughters", "New function - not yet implemented.");
  return 0;
}

inline const char*  TVirtualMCGeometry::VolDaughterName(const char* /*volName*/, Int_t /*i*/) const {
  Warning("VolDaughterName", "New function - not yet implemented.");
  return "";
}

inline Int_t  TVirtualMCGeometry::VolDaughterCopyNo(const char* /*volName*/, Int_t /*i*/) const {
  Warning("VolDaughterCopyNo", "New function - not yet implemented.");
  return 0;
}

#endif //ROOT_TVirtualMCGeometry

