// @(#)root/base:$Id$
// Author: Victor Perev   08/05/02


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "TOffset.h"
#include "TObject.h"
#include "TList.h"
#include "TNamed.h"
#include "TObject.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TMethod.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TROOT.h"

#include <vector>
#include <list>
#include <deque>
#include <map>
#include <set>
//using namespace std ;
using std::vector;
using std::list;
using std::deque;
using std::map;
using std::multimap;
using std::set;
using std::multiset;

    
//  	static members init
#ifndef __CINT__ 
Int_t  TOffset::fgAlign[16]  	= {0};    


Int_t  TOffset::fgSize[16] 	= {0};    
#else 
Int_t  TOffset::fgAlign[16] ;    


Int_t  TOffset::fgSize[16] ;    
#endif
Int_t  TOffset::fgWhereVirt 	= -1;    
Int_t  TOffset::fgSolBug 	= 0;    


//______________________________________________________________________________
 TOffset::TOffset(TClass *cl,Int_t all)
{
  fOffsetList	= new TList;
  fClass  = cl;
  fOffset = fTail = fNBase = fSize = fUsed = fLastMult = 0; 
  Init();
  Virt();
  Adopt(all);
  DoIt();
}
//______________________________________________________________________________
 TOffset::~TOffset()
{
  fOffsetList->Delete();
  delete fOffsetList; fOffsetList=0;
}
//______________________________________________________________________________
const char *TOffset::GetName() const
{
  return (fClass)? fClass->GetName():"";
}

//______________________________________________________________________________
 void TOffset::Init()
{
  typedef vector<int> 			VECTOR;
  typedef list<int>   			LIST;
  typedef deque<int>     		DEQUE;
  typedef map<int,int > 		MAP;
  typedef multimap<int,int > 	        MULTIMAP;
  typedef set<int> 			SET;
  typedef multiset<int > 	        MULTISET;

  class TestChar   	{public: Char_t c; char   	m;};
  class TestShort  	{public: Char_t c; short  	m;};
  class TestInt    	{public: Char_t c; int    	m;};
  class TestLong   	{public: Char_t c; long   	m;};
  class TestPoint  	{public: Char_t c; void*  	m;};
  class TestFloat  	{public: Char_t c; float  	m;};
  class TestDouble 	{public: Char_t c; double 	m;};
  class TestBool   	{public: Char_t c; bool   	m;};
  class TestVector 	{public: Char_t c; VECTOR 	m;};
  class TestList   	{public: Char_t c; LIST   	m;};
  class TestMap    	{public: Char_t c; MAP    	m;};
  class TestMultiMap    {public: Char_t c; MULTIMAP	m;};
  class TestSet    	{public: Char_t c; SET 		m;};
  class TestMultiSet    {public: Char_t c; MULTISET 	m;};

  class TestVirt1  	{public: char c1;};
  class TestVirt2  	{public: virtual void v(){};};
  class TestVirt3 : public TestVirt1,TestVirt2 {public:char c3;};
  class TestVirt4  	{public: virtual void v(){};char c;};

  class sol1Class  {public: double d1[1]; int i1[1];};
  class sol2Class  {public: int i2[1]; };
  class sol3Class : public sol1Class,public sol2Class{};
  class sol4Class : public sol3Class {public: char i4[1];};


 if (fgAlign[kAlignChar]) return;
 fgAlign[kAlignChar] = 1;
 {TestChar   	t; fgAlign[kAlignChar]   	= (char*)&t.m - &t.c;fgSize[kSizeChar]		= sizeof(t.m);}
 {TestShort  	t; fgAlign[kAlignShort]  	= (char*)&t.m - &t.c;fgSize[kSizeShort]		= sizeof(t.m);}
 {TestInt    	t; fgAlign[kAlignInt]    	= (char*)&t.m - &t.c;fgSize[kSizeInt]		= sizeof(t.m);}
 {TestLong   	t; fgAlign[kAlignLong]   	= (char*)&t.m - &t.c;fgSize[kSizeLong]		= sizeof(t.m);}
 {TestPoint  	t; fgAlign[kAlignPoint]  	= (char*)&t.m - &t.c;fgSize[kSizePoint]		= sizeof(t.m);}
 {TestFloat  	t; fgAlign[kAlignFloat]  	= (char*)&t.m - &t.c;fgSize[kSizeFloat]		= sizeof(t.m);}
 {TestDouble 	t; fgAlign[kAlignDouble] 	= (char*)&t.m - &t.c;fgSize[kSizeDouble]	= sizeof(t.m);}
 {TestBool   	t; fgAlign[kAlignBool]   	= (char*)&t.m - &t.c;fgSize[kSizeBool]		= sizeof(t.m);}
 {TestVector   	t; fgAlign[kAlignVector]   	= (char*)&t.m - &t.c;fgSize[kSizeVector]	= sizeof(t.m);}
 {TestList   	t; fgAlign[kAlignList]   	= (char*)&t.m - &t.c;fgSize[kSizeList]		= sizeof(t.m);}
 {TestMap   	t; fgAlign[kAlignMap]   	= (char*)&t.m - &t.c;fgSize[kSizeMap]		= sizeof(t.m);}
 {TestMultiMap	t; fgAlign[kAlignMultiMap]	= (char*)&t.m - &t.c;fgSize[kSizeMultiMap]	= sizeof(t.m);}
 {TestSet   	t; fgAlign[kAlignSet]   	= (char*)&t.m - &t.c;fgSize[kSizeSet]		= sizeof(t.m);}
 {TestMultiSet  t; fgAlign[kAlignMultiSet]   	= (char*)&t.m - &t.c;fgSize[kSizeMultiSet]	= sizeof(t.m);}




 {fgSize[kSizeVirt] = sizeof(TestVirt2);
  TestVirt3  t; fgAlign[kAlignVirt]   = (char*)((TestVirt2*)&t)-&t.c1;}
 {TestVirt4  t; fgWhereVirt = ((char*)&t == &t.c);}
  
//	Test Solaris bug
 { sol4Class t; fgSolBug = ((t.i4-(char*)&t) < (int) sizeof(sol3Class));}
 


  printf(" AlignChar     %d size %d\n",fgAlign[kAlignChar]  	,fgSize[kSizeChar]    );
  printf(" AlignShort    %d size %d\n",fgAlign[kAlignShort] 	,fgSize[kSizeShort]   );
  printf(" AlignInt      %d size %d\n",fgAlign[kAlignInt]   	,fgSize[kSizeInt]     );
  printf(" AlignLong     %d size %d\n",fgAlign[kAlignLong]  	,fgSize[kSizeLong]    );
  printf(" AlignPoint    %d size %d\n",fgAlign[kAlignPoint] 	,fgSize[kSizePoint]   );
  printf(" AlignFloat    %d size %d\n",fgAlign[kAlignFloat] 	,fgSize[kSizeFloat]   );
  printf(" AlignDouble   %d size %d\n",fgAlign[kAlignDouble]	,fgSize[kSizeDouble]  );
  printf(" AlignBool     %d size %d\n",fgAlign[kAlignBool]      ,fgSize[kSizeBool]    );

  printf(" AlignVector   %d size %d\n",fgAlign[kAlignVector]    ,fgSize[kSizeVector]  );
  printf(" AlignList     %d size %d\n",fgAlign[kAlignList]  	,fgSize[kSizeList]    );
  printf(" AlignMap      %d size %d\n",fgAlign[kAlignMap]  	,fgSize[kSizeMap]     );
  printf(" AlignMultiMap %d size %d\n",fgAlign[kAlignMultiMap]	,fgSize[kSizeMultiMap]);
  printf(" AlignSet      %d size %d\n",fgAlign[kAlignSet]      	,fgSize[kSizeSet]     );
  printf(" AlignMultiSet %d size %d\n",fgAlign[kAlignMultiSet] 	,fgSize[kSizeMultiSet]);

  printf(" AlignVirt     %d size %d\n",fgAlign[kAlignVirt]     	,fgSize[kSizeVirt]);
  if (fgWhereVirt==1){
    printf(" VirtTable at the END of class\n");
  } else {
    printf(" VirtTable at the BEGINNING of class\n");}

  if (fgSolBug) printf(" SolarisBug = %d\n",fgSolBug);
}


//______________________________________________________________________________
 void TOffset::Virt()
{
  fVirt = 0;
  TList *lm = fClass->GetListOfMethods();
  if (!lm) return;
  TListIter nextMethod(lm);
  fVirt = fgSize [kSizeVirt ];
  fMult = fgAlign[kAlignVirt];
  TMethod *m =0;
  while ((m = (TMethod*)nextMethod())) {
     Long_t p = m->Property();
     if (p&kIsVirtual) return;
  }
  fVirt = 0;
  fMult = fgAlign[kAlignChar];
}  

//______________________________________________________________________________
 void TOffset::Adopt(Int_t all)
{
 //Adopt info from bases

   TList *lb = fClass->GetListOfBases();
   if (!lb) return; 
   TListIter nextBase(lb);
   TClass *bc=0; 
   TBaseClass *bcl=0;
   while ((bcl = (TBaseClass*)nextBase())) {//list of base
     bc = bcl->GetClassPointer();
     fNBase++; TOffset basOff(bc,all);
     fOffset = Aligne(fOffset,basOff.GetMult());
     fLastMult = basOff.GetMult();
     if (fMult < fLastMult) fMult = fLastMult;
     TList *lo = basOff.GetOffsetList();
     if (!lo) 	continue;
     TListIter nextOff(lo);
     TNamed *tn=0;
     while ((tn=(TNamed*)nextOff()) && all) {	//base offset list
       TString ts(tn->GetTitle());
       ts.Insert(0,"::");
       ts.Insert(0,fClass->GetName());
       TNamed *tnOff = new TNamed(tn->GetName(),ts.Data());
       tnOff->SetUniqueID(tn->GetUniqueID()+fOffset);
       fOffsetList->Add(tnOff);
     }//end of base offset list
     fOffset += basOff.GetUsed();
   }//end list of base
   
   return;
   
}  
//______________________________________________________________________________
 void TOffset::DoIt()
{
   if (fNBase==0 && fgWhereVirt==0) {fOffset = fVirt;}
   TClass *cl= 0;
   TList *ldm = fClass->GetListOfDataMembers();
   TListIter nextDM(ldm);
   TDataMember *dm=0;
   TString FullTypeName,TypeName;
   int kind,size,nMemb=0;
   TNamed *tnoff=0;
   TString tit;
   while((dm=(TDataMember*)nextDM())) {
     Long_t p = dm->Property();
     TDataType *dt = dm->GetDataType();
     FullTypeName = dm->GetFullTypeName();
     tit = fClass->GetName(); tit +=":: type=";
     tit += FullTypeName; 
     TypeName = dm->GetTypeName();
     if (dt) {
       p|=dt->Property();
       FullTypeName = dt->GetFullTypeName();
       TypeName     = dt->GetTypeName();
     }

     if (p&kIsStatic) continue;
     nMemb++;
     int units = 1;
     for (int dim = 0; dim < dm->GetArrayDim(); dim++)
     { units *= dm->GetMaxIndex(dim);}
     tit += " units="; tit+=units;
     
     kind = -1;

     if (p&kIsPointer) 
     { kind = kAlignPoint;}

     else if (p&kIsEnum)
     { kind = kAlignInt;}

     else if (p&kIsFundamental)
     {
       const char *ctypes = "char   short  int    long   void*  float  double bool  ";
//                           12345671234567123456712345671234567123456712345671234567
       TString ty = TypeName.Data();
       ty.ReplaceAll("unsigned ","");
       const char *ctype  = strstr(ctypes,ty.Data());
       assert (ctype);
       kind = (ctype-ctypes)/7;
       
     }
     else if (dm->IsSTLContainer())
     {   
       kind = kAlignVector + dm->IsSTLContainer() - 1;
     }

     if (kind>=0) {
       size = fgSize[kind];
       tit += " size="; tit += size;
       if (fMult < fgAlign[kind]) fMult  = fgAlign[kind];
       tit += " align="; tit += fgAlign[kind];
       fOffset = Aligne(fOffset,fgAlign[kind]);
       tnoff = new TNamed(dm->GetName(),tit.Data());
       tnoff->SetUniqueID(fOffset);
       fOffsetList->AddLast(tnoff);
       fOffset += size*units;
       continue;
     }
       
     cl = gROOT->GetClass(TypeName.Data());
     if (p&kIsClass || cl) {

       assert(cl);
       TOffset offClass(cl);
       fOffset = Aligne(fOffset,offClass.GetMult());
       int virt = offClass.GetVirt();
       if (virt && fMult < fgAlign[kAlignVirt]) fMult  = fgAlign[kAlignVirt];
       tit += " size="; tit += offClass.GetSize();
       tit += " align="; tit += offClass.GetMult();
       tit += " tail="; tit += offClass.GetTail();
       tnoff = new TNamed(dm->GetName(),tit.Data());
       tnoff->SetUniqueID(fOffset);
       fOffsetList->AddLast(tnoff);
       fOffset += (offClass.GetSize())*units;

       


       continue;
     }

     assert(0);
  }

  if (fNBase==0 && fgWhereVirt == 1){	//virt table after class(Linux)
   fOffset = Aligne(fOffset,fgAlign[kAlignVirt]);
   fOffset += fVirt;
  }

  fSize = Aligne(fOffset,fMult);

  if (fgSolBug && nMemb==0 && fNBase > 1) {// Solaris Bug
   fUsed = Aligne(fOffset,fLastMult);
   fTail = (fUsed-fOffset);
   return;
  }

  fUsed = fSize;
  fTail += fSize-fOffset;

}
//______________________________________________________________________________
 void TOffset::Print(const char*) const
{
  printf ("TOffset: Class %s",fClass->GetName());
  printf (" size = %d virt = %d align = %d",fSize,fVirt,fMult);
  printf (" Tail = %d \n",fTail);
  int num = 0;  
  TListIter next(fOffsetList);
  TNamed *tn;
  while ((tn=(TNamed*)next())) {
    printf("%3d -%5d %s\t%s\n",num++,tn->GetUniqueID(),tn->GetName(),tn->GetTitle());
  }
}
//______________________________________________________________________________
Int_t TOffset::GetOffset(const char *name) const
{
  const TObject *tn = fOffsetList->FindObject(name);
  if (!tn) return -1;
  return tn->GetUniqueID();
}


#include "TProfile.h"
#include "TH1.h"
#include "TArrayD.h"

class myProf :public TH1F {
public:
virtual ~myProf(){}
TArrayD fArr;
ClassDef(myProf, 1)
};
class myPro2  {

public:
TH1F fH1;
TArrayD fArr;
};
class myPro3 :public TH1F {
public:
  
virtual ~myPro3(){}
char fArr;
ClassDef(myPro3, 1)
};




//Tail
class fstClass {public:virtual ~fstClass(){};char c0;double d[73]; char c1;};
class tailClass {public:fstClass f; char c2;};

//Tail2
class tail2Class : public fstClass {public: char c2;};
//Tail3
class virtClass  {public:virtual ~virtClass(){};char cv;};
class tail3Class : public fstClass {public: virtClass c2;};

//TailPriv
class fst1Class  {public:virtual ~fst1Class(){}; char c10;double d1[7]; char c11;};
class fst2Class  {public:virtual ~fst2Class(){}; char i2[8]; };
class fst3Class : public fst1Class,fst2Class {public:virtual ~fst3Class(){};};

class tailPClass : public fst3Class {public: char c2;};



//______________________________________________________________________________
void TOffset::Test() 
{

  myProf my;
  printf("size TH1D,TArrayD,my=%zu,%zu,%zu offset=%td\n",
  sizeof(TH1F),sizeof(TArrayD),sizeof(my),(char*)&my.fArr-(char*)&my);

  myPro2 my2;
  printf("size TH1D,TArrayD,my=%zu,%zu,%zu offset=%td\n",
  sizeof(TH1F),sizeof(TArrayD),sizeof(my2),(char*)&my2.fArr-(char*)&my2);

  myPro3 my3;
  printf("size TH1D,TArrayD,my=%zu,%zu,%zu offset=%td\n",
  sizeof(TH1F),sizeof(TArrayD),sizeof(my3),(char*)&my3.fArr-(char*)&my3);

//Tail
{ tailClass t; printf("Tail1: first size=%zu c012offset=%td %td %td,TailSize=%zu\n"
  ,sizeof(fstClass),&t.f.c0-(char*)&t, &t.f.c1-(char*)&t,&t.c2-(char*)&t,sizeof(t));
}
//Tail2
{ tail2Class t; printf("Tail2: first size=%zu c0121offset=%td %td %td,TailSize=%zu\n"
  ,sizeof(fstClass),&t.c0-(char*)&t, &t.c1-(char*)&t, &t.c2-(char*)&t, sizeof(t));
}
//Tail3
{ tail3Class t; printf("Tail3: first size=%zu c012offset=%td %td %td,TailSize=%zu\n"
  ,sizeof(fstClass),&t.c0-(char*)&t, &t.c1-(char*)&t, (char*)&t.c2-(char*)&t, sizeof(t));
}
//Tail4
{ tail2Class t; printf("Tail2: first size=%zu c0121offset=%td %td %td,TailSize=%zu\n"
  ,sizeof(fstClass),&t.c0-(char*)&t, &t.c1-(char*)&t, &t.c2-(char*)&t, sizeof(t));
}
//TailP
{ tailPClass t; printf("TailP: first size=%zu offset=%td TailSize=%zu\n"
  ,sizeof(fst3Class),(char*)&t.c2-(char*)&t, sizeof(t));
}


}


