static int G__ManualTree2_165_6_17(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash)
{
  // We need to emulate
  // return BranchImp(name,classname,TBuffer::GetClass(typeid(T)),addobj,bufsize,splitlevel);

   // Here find the class name 
   G__ClassInfo ti( libp->para[2].tagnum );
   TClass *ptrClass = gROOT->GetClass(ti.Name());
   const char* classname = (const char*)G__int(libp->para[1]);
   TClass *claim = gROOT->GetClass(classname);
   void **addr = (void**)G__int(libp->para[2]);
   
   const char *branchname = (const char*)G__int(libp->para[0]);
   Bool_t error = kFALSE;

   if (ptrClass && claim) {
      if (!(claim->InheritsFrom(ptrClass)||ptrClass->InheritsFrom(claim)) ) {
         // Note we currently do not warning in case of splicing or over-expectation).
         Error("TTree::Branch","The class requested (%s) for the branch \"%s\" is different from the type of the pointer passed (%s)",
               claim->GetName(),branchname,ptrClass->GetName());
         error = kTRUE;
      } else if (addr && *addr) {
         TClass *actualClass = ptrClass->GetActualClass(*addr);
         if (!actualClass) {
            Warning("TTree::Branch", "The actual TClass corresponding to the object provided for the definition of the branch \"%s\" is missing."
                    "\n\tThe object will be truncated down to its %s part",
                    branchname,classname);
         } else if (claim!=actualClass && !actualClass->InheritsFrom(claim)) {
            Error("TTree::Branch", "The actual class (%s) of the object provided for the definition of the branch \"%s\" does not inherit from %s",
                  actualClass->GetName(),branchname,claim->GetName());
            error = kTRUE;
         }
      }
   }
   if (error) {
      G__letint(result7,85,0);   
   } else {
   
      //if (ptrClass) classname = ptrClass->GetName();
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
   TClass *ptrClass = gROOT->GetClass(ti.Name());
   TClass *actualClass = 0;
   void **addr = (void**)G__int(libp->para[1]);
   if (ptrClass && addr) actualClass = ptrClass->GetActualClass(*addr);

   const char *branchname = (const char*)G__int(libp->para[0]);
   if (ptrClass == 0) {
      Error("TTree::Branch","The pointer specified for %s not of a class known to ROOT",
            branchname);
      G__letint(result7,85,0);
   } else {
      const char* classname = ptrClass->GetName();
      if (actualClass==0) {
         Warning("TTree::Branch", "The actual TClass corresponding to the object provided for the definition of the branch \"%s\" is missing."
            "\n\tThe object will be truncated down to its %s part",
                 branchname,classname);
      } else {
         classname = actualClass->GetName();
      }
      TTree *t = (TTree*)(G__getstructoffset());
      switch(libp->paran) {
      case 4:
         G__letint(result7,85,(long)t->Branch(branchname,
                                              classname,
                                              (void*)G__int(libp->para[1]),
                                              (Int_t)G__int(libp->para[2]),
                                              (Int_t)G__int(libp->para[3])));
         break;
      case 3:
         G__letint(result7,85,(long)t->Branch(branchname,
                                              classname,
                                              (void*)G__int(libp->para[1]),
                                              (Int_t)G__int(libp->para[2])));
         break;
      case 2:
         G__letint(result7,85,(long)t->Branch(branchname,
                                              classname,
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
   TClass *ptrClass = gROOT->GetClass(type.c_str());
   TDataType *data = gROOT->GetType(type.c_str());
   EDataType datatype = data ? (EDataType)data->GetType() : kOther_t;

   ((TTree*)(G__getstructoffset()))->SetBranchAddress((const char*)G__int(libp->para[0]),(void*)G__int(libp->para[1]),ptrClass,datatype,ti.Reftype()==G__PARAP2P);
   return(1 || funcname || hash || result7 || libp) ;
}
