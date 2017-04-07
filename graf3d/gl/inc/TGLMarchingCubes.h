// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  06/01/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLMarchingCubes
#define ROOT_TGLMarchingCubes

#include <vector>

#include "TH3.h"

#include "TGLIsoMesh.h"
#include "TKDEAdapter.h"

/*
Implementation of "marching cubes" algortihm for GL module. Used by
TGLTF3Painter and TGLIsoPainter.
Good and clear algorithm explanation can be found here:
http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
*/

class TF3;
class TKDEFGT;

namespace Rgl {
namespace Mc {

/*
Some routines, values and tables for marching cube method.
*/
extern const UInt_t  eInt[256];
extern const Float_t vOff[8][3];
extern const UChar_t eConn[12][2];
extern const Float_t eDir[12][3];
extern const Int_t   conTbl[256][16];

/*
"T" prefix in class names is only for code-style checker.
*/

/*
TCell is a cube from marching cubes algorithm.
It has "type" - defines, which vertices
are under iso level, which are above.

Vertices numeration:

           |z
           |
           4____________7
          /|           /|
         / |          / |
        /  |         /  |
       /   |        /   |
      5____|_______6    |
      |    0_______|____3______ y
      |   /        |   /
      |  /         |  /
      | /          | /
      |/           |/
      1____________2
     /
    /x

TCell's "type" is 8-bit number, one bit per vertex.
So, if vertex 1 and 2 are under iso-surface, type
will be:

 7 6 5 4 3 2 1 0 (bit number)
[0 0 0 0 0 1 1 0] bit pattern

type == 6.

Edges numeration:

           |z
           |
           |_____7______
          /|           /|
         / |          / |
       4/  8         6  11
       /   |        /   |
      /____|5______/    |
      |    |_____3_|____|______ y
      |   /        |   /
      9  /        10  /
      | /0         | /2
      |/           |/
      /____________/
     /      1
    /x

There are 12 edges, any of them can be intersected by iso-surface
(not all 12 simultaneously). Edge's intersection is a vertex in
iso-mesh's vertices array, cell holds index of this vertex in
fIds array.
fVals holds "scalar field" or density values in vertices [0, 7].

"V" parameter is the type to hold such values.
*/

template<class V>
class TCell {
public:
   TCell() : fType(), fIds(), fVals()
   {
      //TCell ctor.
      //Such mem-initializer list can produce
      //warnings with some versions of MSVC,
      //but this list is what I want.
   }

   UInt_t     fType;
   UInt_t     fIds[12];
   V          fVals[8];
};

/*
TSlice of marching cubes' grid. Has W * H cells.
If you have TH3 hist, GetNbinsX() is W and GetNbinsY() is H.
*/
template<class V>
class TSlice {
public:
   TSlice()
   {
   }

   void ResizeSlice(UInt_t w, UInt_t h)
   {
      fCells.resize(w * h);
   }

   std::vector<TCell<V> > fCells;
private:
   TSlice(const TSlice &rhs);
   TSlice & operator = (const TSlice &rhs);
};

/*
Mesh builder requires generic "data source": it can
be a wrapped TH3 object, a wrapped TF3 object or some
"density estimator" object.
Mesh builder inherits this data source type.

TH3Adapter is one of such data sources.
It has _direct_ access to TH3 internal data.
GetBinContent(i, j, k) is a virtual function
and it calls two other virtual functions - this
is very expensive if you call GetBinContent
several million times as I do in marching cubes.

"H" parameter is one of TH3 classes,
"E" is the type of internal data.

For example, H == TH3C, E == Char_t.
*/

template<class H, class E>
class TH3Adapter {
protected:

   typedef E    ElementType_t;

   TH3Adapter()
      : fSrc(0), fW(0), fH(0), fD(0), fSliceSize(0)
   {
   }

   UInt_t GetW()const
   {
      return fW - 2;
   }

   UInt_t GetH()const
   {
      return fH - 2;
   }

   UInt_t GetD()const
   {
      return fD - 2;
   }

   void SetDataSource(const H *hist)
   {
      fSrc = hist->GetArray();
      fW   = hist->GetNbinsX() + 2;
      fH   = hist->GetNbinsY() + 2;
      fD   = hist->GetNbinsZ() + 2;
      fSliceSize = fW * fH;
   }

   void FetchDensities()const{}//Do nothing.

   ElementType_t GetData(UInt_t i, UInt_t j, UInt_t k)const
   {
      i += 1;
      j += 1;
      k += 1;
      return fSrc[k * fSliceSize + j * fW + i];
   }

   const ElementType_t *fSrc;
   UInt_t fW;
   UInt_t fH;
   UInt_t fD;
   UInt_t fSliceSize;
};

/*
TF3Adapter. Lets TMeshBuilder to use TF3 as a 3d array.
TF3Adapter, TF3EdgeSplitter (see below) and TMeshBuilder<TF3, ValueType>
need TGridGeometry<ValueType>, so TGridGeometry is a virtual base.
*/

class TF3Adapter : protected virtual TGridGeometry<Double_t> {
protected:
   typedef Double_t ElementType_t;

   TF3Adapter() : fTF3(0), fW(0), fH(0), fD(0)
   {
   }

   UInt_t GetW()const
   {
      return fW;
   }

   UInt_t GetH()const
   {
      return fH;
   }

   UInt_t GetD()const
   {
      return fD;
   }

   void SetDataSource(const TF3 *f);

   void FetchDensities()const{}//Do nothing.

   Double_t GetData(UInt_t i, UInt_t j, UInt_t k)const;

   const TF3 *fTF3;//TF3 data source.
   //TF3 grid's dimensions.
   UInt_t     fW;
   UInt_t     fH;
   UInt_t     fD;
};

/*
TSourceAdapterSelector is aux. class used by TMeshBuilder to
select "data-source" base depending on data-source type.
*/
template<class> class TSourceAdapterSelector;

template<>
class TSourceAdapterSelector<TH3C> {
public:
   typedef TH3Adapter<TH3C, Char_t> Type_t;
};

template<>
class TSourceAdapterSelector<TH3S> {
public:
   typedef TH3Adapter<TH3S, Short_t> Type_t;
};

template<>
class TSourceAdapterSelector<TH3I> {
public:
   typedef TH3Adapter<TH3I, Int_t> Type_t;
};

template<>
class TSourceAdapterSelector<TH3F> {
public:
   typedef TH3Adapter<TH3F, Float_t> Type_t;
};

template<>
class TSourceAdapterSelector<TH3D> {
public:
   typedef TH3Adapter<TH3D, Double_t> Type_t;
};

template<>
class TSourceAdapterSelector<TF3> {
public:
   typedef TF3Adapter Type_t;
};

template<>
class TSourceAdapterSelector<TKDEFGT> {
public:
   typedef Fgt::TKDEAdapter Type_t;
};

/*
Edge splitter is the second base class for TMeshBuilder.
Its task is to split cell's edge by adding new vertex
into mesh.
Default splitter is used by TH3 and KDE.
*/

template<class E, class V>
V GetOffset(E val1, E val2, V iso)
{
   const V delta = val2 - val1;
   if (!delta)
      return 0.5f;
   return (iso - val1) / delta;
}

template<class H, class E, class V>
class TDefaultSplitter : protected virtual TGridGeometry<V> {
protected:
   void SetNormalEvaluator(const H * /*source*/)
   {
   }
   void SplitEdge(TCell<E> & cell, TIsoMesh<V> * mesh, UInt_t i,
                  V x, V y, V z, V iso)const
{
   V v[3];
   const V offset = GetOffset(cell.fVals[eConn[i][0]],
                              cell.fVals[eConn[i][1]],
                              iso);
   v[0] = x + (vOff[eConn[i][0]][0] + offset * eDir[i][0]) * this->fStepX;
   v[1] = y + (vOff[eConn[i][0]][1] + offset * eDir[i][1]) * this->fStepY;
   v[2] = z + (vOff[eConn[i][0]][2] + offset * eDir[i][2]) * this->fStepZ;
   cell.fIds[i] = mesh->AddVertex(v);
}


};

/*
TF3's edge splitter. Calculates new vertex and surface normal
in this vertex using TF3.
*/

class TF3EdgeSplitter : protected virtual TGridGeometry<Double_t> {
protected:
   TF3EdgeSplitter() : fTF3(0)
   {
   }

   void SetNormalEvaluator(const TF3 *tf3)
   {
      fTF3 = tf3;
   }

   void SplitEdge(TCell<Double_t> & cell, TIsoMesh<Double_t> * mesh, UInt_t i,
                  Double_t x, Double_t y, Double_t z, Double_t iso)const;

   const TF3 *fTF3;
};

/*
TSplitterSelector is aux. class to select "edge-splitter" base
for TMeshBuilder.
*/

template<class, class> class TSplitterSelector;

template<class V>
class TSplitterSelector<TH3C, V> {
public:
   typedef TDefaultSplitter<TH3C, Char_t, V> Type_t;
};

template<class V>
class TSplitterSelector<TH3S, V> {
public:
   typedef TDefaultSplitter<TH3S, Short_t, V> Type_t;
};

template<class V>
class TSplitterSelector<TH3I, V> {
public:
   typedef TDefaultSplitter<TH3I, Int_t, V> Type_t;
};

template<class V>
class TSplitterSelector<TH3F, V> {
public:
   typedef TDefaultSplitter<TH3F, Float_t, V> Type_t;
};

template<class V>
class TSplitterSelector<TH3D, V> {
public:
   typedef TDefaultSplitter<TH3D, Double_t, V> Type_t;
};

template<class V>
class TSplitterSelector<TKDEFGT, V> {
public:
   typedef TDefaultSplitter<TKDEFGT, Float_t, Float_t> Type_t;
};

template<class V>
class TSplitterSelector<TF3, V> {
public:
   typedef TF3EdgeSplitter Type_t;
};

/*
Mesh builder. Polygonizes scalar field - TH3, TF3 or
something else (some density estimator as data-source).

ValueType is Float_t or Double_t - the type of vertex'
x,y,z components.
*/

template<class DataSource, class ValueType>
class TMeshBuilder : public TSourceAdapterSelector<DataSource>::Type_t,
                     public TSplitterSelector<DataSource, ValueType>::Type_t
{
private:
   //Two base classes.
   typedef typename TSourceAdapterSelector<DataSource>::Type_t       DataSourceBase_t;
   typedef typename TSplitterSelector<DataSource, ValueType>::Type_t SplitterBase_t;
   //Using declarations required, since these are
   //type-dependant names in template.
   using DataSourceBase_t::GetW;
   using DataSourceBase_t::GetH;
   using DataSourceBase_t::GetD;
   using DataSourceBase_t::GetData;
   using SplitterBase_t::SplitEdge;

   typedef typename DataSourceBase_t::ElementType_t ElementType_t;

   typedef TCell<ElementType_t>  CellType_t;
   typedef TSlice<ElementType_t> SliceType_t;
   typedef TIsoMesh<ValueType>   MeshType_t;

public:
   TMeshBuilder(Bool_t averagedNormals, ValueType eps = 1e-7)
      : fAvgNormals(averagedNormals), fMesh(0), fIso(), fEpsilon(eps)
   {
   }

   void BuildMesh(const DataSource *src, const TGridGeometry<ValueType> &geom,
                  MeshType_t *mesh, ValueType iso);

private:

   Bool_t      fAvgNormals;
   SliceType_t fSlices[2];
   MeshType_t *fMesh;
   ValueType   fIso;
   ValueType   fEpsilon;

   void NextStep(UInt_t depth, const SliceType_t *prevSlice,
                 SliceType_t *curr)const;

   void BuildFirstCube(SliceType_t *slice)const;
   void BuildRow(SliceType_t *slice)const;
   void BuildCol(SliceType_t *slice)const;
   void BuildSlice(SliceType_t *slice)const;
   void BuildFirstCube(UInt_t depth, const SliceType_t *prevSlice,
                       SliceType_t *slice)const;
   void BuildRow(UInt_t depth, const SliceType_t *prevSlice,
                 SliceType_t *slice)const;
   void BuildCol(UInt_t depth, const SliceType_t *prevSlice,
                 SliceType_t *slice)const;
   void BuildSlice(UInt_t depth, const SliceType_t *prevSlice,
                   SliceType_t *slice)const;

   void BuildNormals()const;

   TMeshBuilder(const TMeshBuilder &rhs);
   TMeshBuilder & operator = (const TMeshBuilder &rhs);
};

}//namespace Mc
}//namespace Rgl

#endif
