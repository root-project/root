// tutorial illustrating the use of TMath::Permute
//  can be run with:
// root > .x permute.C
// root > .x permute.C+ with ACLIC
      
#include <TMath.h>

int permuteSimple () 
{
  printf("\nTMath::Permute simple test\n");
  printf("==========================\n");
  Int_t a[4];
  Bool_t start=kTRUE;
  for(Int_t k=0;k<24;k++) {
    TMath::Permute(a,4,start);
    for(Int_t i=0;i<4;printf("%c",97+a[i++])); printf("\n");
  }
  return 0;
}


Int_t permuteFancy() 
{
  Int_t a[10];
  Int_t &n=a[0], &i=a[1];
  Int_t &e=a[2], &t=a[3];
  Int_t &h=a[4], &r=a[5];
  Int_t &f=a[6], &o=a[7];
  Int_t &s=a[8], &u=a[9];
  Int_t nine, three, neuf, trois;
  Bool_t start=kTRUE;

  printf("\nTMath::Permute fancy test\n");
  printf("=========================\n");
  printf("This is a program to calculate the solution to the following problem\n");
  printf("Find the equivalence between letters and numbers so that\n\n");
  printf("              NINE*THREE = NEUF*TROIS\n\n");
  while(TMath::Permute(a,10,start)) {
    nine=((n*10+i)*10+n)*10+e;
    neuf=((n*10+e)*10+u)*10+f;
    three=(((t*10+h)*10+r)*10+e)*10+e;
    trois=(((t*10+r)*10+o)*10+i)*10+s;
    if(nine*three==neuf*trois) {
      printf("Solution found!\n\n");
      printf("T=%d N=%d E=%d S=%d F=%d H=%d R=%d I=%d O=%d U=%d\n",t,n,e,s,f,h,r,i,o,u);
      printf("NINE=%d THREE=%d NEUF=%d TROIS=%d\n",nine,three,neuf,trois);
      printf("NINE*THREE = NEUF*TROIS = %d\n",neuf*trois);
      return 0;
    }
  }
  printf("No solutions found -- something is wrong here!\n");
  return 0;
}

void permute() {
   permuteSimple();
   permuteFancy();
}
