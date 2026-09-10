## \file
## \ingroup tutorial_math
## \notebook -nodraw
##
## Tutorial illustrates the use of TMath.Permute function.
## The script can be run with:
##
## ~~~{.cpp}
## IP[0] : %run permute.py
## ~~~
##
## \macro_output
## \macro_code
##
## \author Federico Carminati
## \translator P. P.


import ROOT
import ctypes

#classes
TMath = ROOT.TMath 

#math
Math = ROOT.Math

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t
c_int = ctypes.c_int
c_double = ctypes.c_double

#utils
def printf(string, *args):
   return print(string % args, end="")

def to_c(ls):
#def to_c(ls, Type=Int_t):
   #if Type == Int_t:
   #   return ( c_int * len(ls) )( * ls )
   #if Type == Double_t:
   #   return ( c_double * len(ls) )( * ls )

   # or maybe
   if isinstance(ls[0], int):
      return ( c_int * len(ls) )( * ls )
   if isinstance(ls[0], float):
      return ( c_double * len(ls) )( * ls )

# int
def permuteSimple1() :
   printf("\nTMath.Permute simple test\n")
   printf("==========================\n")

   aa = chr( ord('a') )
   a = [Int_t() for _ in range(4) ]
   a = to_c(a)
   
   i = Int_t(0) 
   icount = 0

   #for (i = 0; i < 4; i++)
   for i in range(0, 4, 1):
      a[i] = i

   # do {
   while( True ):
      icount += 1
      #for( i = 0; i < 4; i++) 
      #BP:
      #DOING:
      #for i in range(0, 4, 1): printf("%c", chr( ord(aa) + a[i] ) )
      i = 0
      while ( i < 4 ):
         printf("%c", chr( ord(aa) + a[i] ) )
         i += 1
      
      printf("\n")
   #}
   #while(
      if not TMath.Permute(4, a):
         break   
   #)
   printf("Found %d permutations = 4!\n", icount)
   return 0
   

# int
def permuteSimple2() :
   printf("\nTMath.Permute simple test with repetition\n")
   printf("==========================================\n")

   aa = chr( ord('a') - 1)
   a = [Int_t() for _ in range(6)] 
   a = to_c(a)
   
   i = Int_t()
   icount = 0
   #for (i = 0; i < 6; i++)
   for i in range(0, 6, 1):
      a[i] = Int_t( (i + 2) / 2 )
   #do {
   while True:
      icount += 1

      #for(i = 0; i < 5; printf("%c", static_cast<char>(aa + a[i++])))
      #for i in range(0, 5, 1): printf("%c", chr( ord(aa) + a[i] ) )
      i = 0
      while ( i < 5 ): 
         printf("%c", chr( ord(aa) + a[i] ) )
         i += 1 
      
      printf("\n")
   #}
   #while(
      if not TMath.Permute(5, a): 
         break   
   #)
   printf("Found %d permutations = 5!/(2! 2!)\n", icount)
   return 0
   

# Int_t
def permuteFancy() :
   a = [Int_t() ] * 10
   a = to_c(a)

   def pass_by_reference(a):
      n = a[0] ; i = a[1]
      e = a[2] ; t = a[3]
      h = a[4] ; r = a[5]
      f = a[6] ; o = a[7]
      s = a[8] ; u = a[9]
      return n, i, e, t, h, r, f ,o, s, u

   n, i, e, t, h, r, f ,o, s, u = pass_by_reference(a)


   nine, three, neuf, trois = [ Int_t() for _ in range(4) ]
   
   printf("\nTMath.Permute fancy test\n")
   printf("=========================\n")
   printf("This is a program to calculate the solution to the following problem\n")
   printf("Find the equivalence between letters and numbers so that\n\n")
   printf("              NINE*THREE = NEUF*TROIS\n\n")

   #for (ii = 0; ii < 10; ii++)
   for ii in range(0, 10, 1):
      a[ii] = ii

   #If not pass_by_reference, the solution is trivial. 0,0,0,0 == 0,0,0,0
   n, i, e, t, h, r, f ,o, s, u = pass_by_reference(a)

   #do {
   while True: 
      n, i, e, t, h, r, f ,o, s, u = pass_by_reference(a)

      nine = ((n * 10 + i) * 10 + n) * 10 + e
      neuf = ((n * 10 + e) * 10 + u) * 10 + f
      three = (((t * 10 + h) * 10 + r) * 10 + e) * 10 + e
      trois = (((t * 10 + r) * 10 + o) * 10 + i) * 10 + s
      if (nine*three==neuf*trois) :
         printf("Solution found!\n\n")
         printf("T=%d N=%d E=%d S=%d F=%d H=%d R=%d I=%d O=%d U=%d\n", t, n, e, s,
                f, h, r, i, o, u)
         printf("NINE=%d THREE=%d NEUF=%d TROIS=%d\n", nine, three, neuf, trois)
         printf("NINE*THREE = NEUF*TROIS = %d\n", neuf * trois)
         return 0
      #}
      #while ( 
      if not TMath.Permute(10, a):
         break   
      #)
   printf("No solutions found -- something is wrong here!\n")
   return 0
   

# void
def permute() :
   permuteSimple1()
   permuteSimple2()
   permuteFancy()
   


if __name__ == "__main__":
   permute()
