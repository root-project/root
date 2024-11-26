/// \file
/// \ingroup tutorial_math
/// \notebook -nodraw
/// Tutorial illustrating the use of TMath::Permute
/// can be run with:
///
/// ~~~{.cpp}
/// root > .x permute.C
/// root > .x permute.C+ with ACLIC
/// ~~~
///
/// \macro_output
/// \macro_code
///
/// \author Federico Carminati

#include <TMath.h>

int permuteSimple1 ()
{
  printf("\nTMath::Permute simple test\n");
  printf("==========================\n");
  char aa='a';
  Int_t a[4];
  Int_t i;
  Int_t icount=0;
  for(i=0; i<4; i++) a[i]=i;
  do {
    icount++;
    for(Int_t i=0;i<4;printf("%c",static_cast<char>(aa+a[i++])));
    printf("\n");

  } while(TMath::Permute(4,a));
  printf("Found %d permutations = 4!\n",icount);
  return 0;
}

int permuteSimple2 ()
{
  printf("\nTMath::Permute simple test with repetition\n");
  printf("==========================================\n");
  char aa='a'-1;
  Int_t a[6];
  Int_t i;
  Int_t icount=0;
  for(i=0; i<6; i++) a[i]=(i+2)/2;
  do {
     icount++;
     for(Int_t i=0;i<5;printf("%c",static_cast<char>(aa+a[i++])));
     printf("\n");

  } while(TMath::Permute(5,a));
  printf("Found %d permutations = 5!/(2! 2!)\n",icount);
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

  printf("\nTMath::Permute fancy test\n");
  printf("=========================\n");
  printf("This is a program to calculate the solution to the following problem\n");
  printf("Find the equivalence between letters and numbers so that\n\n");
  printf("              NINE*THREE = NEUF*TROIS\n\n");
  for(Int_t ii=0; ii<10; ii++) a[ii]=ii;
  do {
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
  } while(TMath::Permute(10,a));
  printf("No solutions found -- something is wrong here!\n");
  return 0;
}

void permute() {
   permuteSimple1();
   permuteSimple2();
   permuteFancy();
}
