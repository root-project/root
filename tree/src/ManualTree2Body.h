static int G__ManualTree2_165_6_17(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
  // We need to emulate
  // return BranchImp(name,classname,TBuffer::GetClass(typeid(T)),addobj,bufsize,splitlevel);

   // Here find the class name 
   G__ClassInfo ti( libp->para[2].tagnum );
   TClass *realClass = gROOT->GetClass(ti.Name());
   const char* classname = (const char*)G__int(libp->para[1]);
   TClass *claim = gROOT->GetClass(classname);
   
   const char *branchname = (const char*)G__int(libp->para[0]);
   if (realClass && claim && !(claim->InheritsFrom(realClass)||realClass->InheritsFrom(claim)) ) {
      // Note we currently do not warning in case of splicing or over-expectation).
      Error("TTree::Branch","The class requested (%s) for the branch \"%s\" is different from the type of the pointer passed (%s)",
            claim->GetName(),branchname,realClass->GetName());
      G__letint(result7,85,0);
   } else {
   
      if (realClass) classname = realClass->GetName();
      TTree *t = (TTree*)(G__getstructoffset());
      switch(libp->paran) {
      case 5:
         G__letint(result7,85,(long)t->Branch(branchname,
                                              classname,
                                              (void*)G__int(libp->para[2]),
                                              (Int_t)G__int(libp->para[3]),
                                              (Int_t)G__int(libp->para[4])));
         break;
      case 4:
         G__letint(result7,85,(long)t->Branch(branchname,
                                              classname,
                                              (void*)G__int(libp->para[2]),
                                              (Int_t)G__int(libp->para[3])));
         break;
      case 3:
         G__letint(result7,85,(long)t->Branch(branchname,
                                              classname,
                                              (void*)G__int(libp->para[2])));
         break;
      }
   }
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ManualTree2_165_7_17(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   // We need to emulate 
   // return BranchImp(name,TBuffer::GetClass(typeid(T)),addobj,bufsize,splitlevel);

   G__ClassInfo ti( libp->para[1].tagnum );
   TClass *realClass = gROOT->GetClass(ti.Name());

   const char *branchname = (const char*)G__int(libp->para[0]);
   if (realClass == 0) {
      Error("TTree::Branch","The pointer specified for %s not of a class known to ROOT",
            branchname);
      G__letint(result7,85,0);
   } else {
      TTree *t = (TTree*)(G__getstructoffset());
      switch(libp->paran) {
      case 4:
         G__letint(result7,85,(long)t->Branch(branchname,
                                              realClass->GetName(),
                                              (void*)G__int(libp->para[1]),
                                              (Int_t)G__int(libp->para[2]),
                                              (Int_t)G__int(libp->para[3])));
         break;
      case 3:
         G__letint(result7,85,(long)t->Branch(branchname,
                                              realClass->GetName(),
                                              (void*)G__int(libp->para[1]),
                                              (Int_t)G__int(libp->para[2])));
         break;
      case 2:
         G__letint(result7,85,(long)t->Branch(branchname,
                                              realClass->GetName(),
                                              (void*)G__int(libp->para[1])));
         break;
      }         
   }
   return(1 || funcname || hash || result7 || libp) ;
}

#include "TDataType.h"

static int G__ManualTree2_165_9_17(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
   G__setnull(result7);

   G__TypeInfo ti( libp->para[1] );
   string type( TClassEdit::ShortType(ti.Name(),TClassEdit::kDropTrailStar) );
   TClass *realClass = gROOT->GetClass(type.c_str());
   TDataType *data = gROOT->GetType(type.c_str());
   EDataType datatype = data ? (EDataType)data->GetType() : kOther_t;

   ((TTree*)(G__getstructoffset()))->SetBranchAddress((const char*)G__int(libp->para[0]),(void*)G__int(libp->para[1]),realClass,datatype,ti.Reftype()==G__PARAP2P);
   return(1 || funcname || hash || result7 || libp) ;
}
