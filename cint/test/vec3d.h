/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// vec3d.h ///////////////////////////////////////////////////////////

// 2 dimentional array
template<class T> class array2d {
 public:
  array2d(int jx,int kx) { 
    maxj=jx; maxk=kx;
    isnew=1;
    p = new T[maxj*maxk];
  }
  array2d(T* pin,int jx,int kx) { 
    maxj=jx; maxk=kx;
    isnew=0;
    p = pin;
  }
  ~array2d() {if(isnew) delete[] p;}
  T& operator()(int j,int k) {return( *(p + maxk*j + k) ) ;}
  T* operator[](int j) {return(p+maxk*j);}
 private:
  T* p;
  int maxj,maxk;
  int isnew;
};

// 3 dimentional array
template<class T> class array3d {
 public:
  array3d(int ix,int jx,int kx) { 
    maxi=ix; maxj=jx; maxk=kx;
    isnew=1;
    p = new T[maxi*maxj*maxk];
  }
  array3d(T* pin,int ix,int jx,int kx) { 
    maxi=ix; maxj=jx; maxk=kx;
    isnew=0;
    p = pin;
  }
  ~array3d() {if(isnew) delete[] p;}
  T& operator()(int i,int j,int k) {
    return( *(p + ((maxj*maxk)*i + maxk*j + k)) );
  }
  array2d<T> operator[](int i) {
    array2d<T> tmp(p+maxj*maxk*i,maxj,maxk);
    return(tmp);
  }
 private:
  T* p;
  int maxi,maxj,maxk;
  int isnew;
};

typedef array3d<double> double3d;
typedef array2d<double> double2d;
typedef array3d<float> float3d;
typedef array2d<float> float2d;
typedef array3d<int> int3d;
typedef array2d<int> int2d;


