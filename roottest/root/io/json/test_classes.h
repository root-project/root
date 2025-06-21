#ifndef test_classes_h
#define test_classes_h

#include <string>
#include <vector>
#include <list>
#include <deque>
#include <map>
#include <set>
#include <bitset>
#include <iostream>
#include <cstring>

#include "TNamed.h"
#include "TBox.h"
#include "TObjString.h"
#include "TList.h"
#include "TObjArray.h"
#include "TMap.h"
#include "TClonesArray.h"
#include "TArrayI.h"
#include "TBufferJSON.h"
#include "TClass.h"

class TJsonEx1 {
   protected:
     bool       fBool{false};
     char       fChar{0};
     short      fShort{0};
     int        fInt{0};      // *OPTION={SetMethod="SetI";GetMethod="GetI"}
     long       fLong{0};     // *OPTION={SetMethod="SetLong";GetMethod="GetLong"}
     float      fFloat{0};
     double     fDouble{0};

   public:
     int        GetI() { return fInt; }
     long       GetLong() { return fLong; }

     void       SetI(int zn) { fInt = zn; }
     void       SetLong(long zn) { fLong = zn; }

     TJsonEx1() = default;
     virtual ~TJsonEx1() {}

     void Init(int n = 1)
     {
        fBool = n % 2;
        fChar = 'C';
        fShort = n*123;
        fInt = n*123456;
        fLong = n*7654321;
        fFloat = n*1.2;
        fDouble = n*3.8;
     }

     bool operator<(const TJsonEx1& ex) const {
        return fInt < ex.fInt;
     }


     virtual void Print()
     {
         std::cout << "   fBool = " << fBool << std::endl;
         std::cout << "   fChar = " << fChar << std::endl;
         std::cout << "   fShort = " << fShort << std::endl;
         std::cout << "   fInt = " << fInt << std::endl;
         std::cout << "   fLong = " << fLong << std::endl;
         std::cout << "   fFloat = " << fFloat << std::endl;
         std::cout << "   fDouble = " << fDouble << std::endl;
     }

};

class TJsonEx11 : public TJsonEx1 {
   public:

      int        fInt2;

      TJsonEx11() : TJsonEx1(), fInt2(0) {
      }

      void Init(int z = 1)
      {
         TJsonEx1::Init(z);
         fInt2 = 27*z;
      }

};

// _______________________________________________________________

class TJsonEx2 {
   protected:
     int        fTest1[4];
     int        fTest2[2][2];
     bool       fBool[2][3][4];
     char       fChar[2][3][4];
     short      fShort[2][3][4];
     int        fInt[2][3][4];
     long       fLong[2][3][4];
     float      fFloat[2][3][4];
     double     fDouble[2][3][4];
   public:
     TJsonEx2()
     {
        fTest1[0] = 0;
        fTest1[1] = 0;
        fTest1[2] = 0;
        fTest1[3] = 0;

        fTest2[0][0] = 0;
        fTest2[0][1] = 0;
        fTest2[1][0] = 0;
        fTest2[1][1] = 0;

        for (int i=0;i<2;i++)
          for (int j=0;j<3;j++)
            for (int k=0;k<4;k++) {
               fBool[i][j][k] = 0;
               fChar[i][j][k] = 0;
               fShort[i][j][k] = 0;
               fInt[i][j][k] = 0;
               fLong[i][j][k] = 0;
               fFloat[i][j][k] = 0;
               fDouble[i][j][k] = 0;
            }
     }

     virtual ~TJsonEx2() {}

     void Init(int zz = 1)
     {
        fTest1[0] = zz*11111;
        fTest1[1] = zz*22222;
        fTest1[2] = zz*33333;
        fTest1[3] = zz*44444;

        fTest2[0][0] = zz*1;
        fTest2[0][1] = zz*2;
        fTest2[1][0] = zz*3;
        fTest2[1][1] = zz*4;

        for (int i=0;i<2;i++)
          for (int j=0;j<3;j++)
            for (int k=0;k<4;k++) {
               fBool[i][j][k] = ((i+j+k) % 2) !=0;
               fChar[i][j][k] = 48 + i+j+k;
               fShort[i][j][k] = i+j+k+zz;
               fInt[i][j][k] = (i+1)*(j+2)+k+zz;
               fLong[i][j][k] = (i+1)*(j+2)*(k+3)*zz;
               fFloat[i][j][k] = (i+4)*(j+3)*(k+2)*zz;
               fDouble[i][j][k] = (i+1)*(j+5)*(k+9)*zz;
            }
     }

     void Print()
     {
         for (int i=0;i<4;i++)
            std::cout << "   fTest1[" << i << "] = " << fTest1[i] << std::endl;

         for (int i=0;i<2;i++)
         for (int j=0;j<3;j++)
         for (int k=0;k<4;k++) {
            std::cout << "  for indexes ["<<i<<"]["<<j<<"]["<<k<<"]" << std::endl;
            std::cout << "     fBool = " << fBool[i][j][k] << std::endl;
            std::cout << "     fChar = " << fChar[i][j][k] << std::endl;
            std::cout << "     fShort = " << fShort[i][j][k] << std::endl;
            std::cout << "     fInt = " << fInt[i][j][k] << std::endl;
            std::cout << "     fLong = " << fLong[i][j][k] << std::endl;
            std::cout << "     fFloat = " << fFloat[i][j][k] << std::endl;
            std::cout << "     fDouble = " << fDouble[i][j][k] << std::endl;
         }
     }
};

// _______________________________________________________________

class TJsonEx3 {
   protected:
     int        fSize{0};
     bool       *fBool{nullptr};     // [fSize]
     char       *fChar{nullptr};     // [fSize]
     short      *fShort{nullptr};    // [fSize]
     int        *fInt{nullptr};      // [fSize]
     long       *fLong{nullptr};     // [fSize]
     float      *fFloat{nullptr};    // [fSize]
     double     *fDouble{nullptr};   // [fSize]
   public:

     TJsonEx3(int sz = 0)
     {
        if (sz>0) Init(sz);
     }

     TJsonEx3(const TJsonEx3 &src)
     {
        Allocate(src.fSize);
        for (int n=0;n<fSize;n++) {
           fBool[n] = src.fBool[n];
           fChar[n] = src.fChar[n];
           fShort[n] = src.fShort[n];
           fInt[n] = src.fInt[n];
           fLong[n] = src.fLong[n];
           fFloat[n] = src.fFloat[n];
           fDouble[n] = src.fDouble[n];
        }

     }

     virtual ~TJsonEx3()
     {
        Release();
     }

     void Allocate(int sz)
     {
        Release();
        if (sz<=0) return;

        fSize = sz;
        fBool = new bool[fSize];
        fChar = new char[fSize];
        fShort = new short[fSize];
        fInt = new int[fSize];
        fLong = new long[fSize];
        fFloat = new float[fSize];
        fDouble = new double[fSize];
     }

     void Release()
     {
       if (fSize<=0) return;
       delete [] fBool; fBool = nullptr;
       delete [] fChar; fChar = nullptr;
       delete [] fShort; fShort = nullptr;
       delete [] fInt; fInt = nullptr;
       delete [] fLong; fLong = nullptr;
       delete [] fFloat; fFloat = nullptr;
       delete [] fDouble; fDouble = nullptr;
       fSize = 0;
     }

     void Init(int sz = 7)
     {
        Allocate(sz);

        for (int n=0;n<sz;n++) {
           fBool[n] = false;
           fChar[n] = 49 + n;
           fShort[n] = n*2;
           fInt[n] = n*5;
           fLong[n] = n*123;
           fFloat[n] = n*3;
           fDouble[n] = n*7;
        }
     }


     void Print()
     {
         for (int i=0;i<fSize;i++) {
            std::cout << "  index = " << i << std::endl;
            std::cout << "     fBool = " << fBool[i] << std::endl;
            std::cout << "     fChar = " << fChar[i] << std::endl;
            std::cout << "     fShort = " << fShort[i] << std::endl;
            std::cout << "     fInt = " << fInt[i] << std::endl;
            std::cout << "     fLong = " << fLong[i] << std::endl;
            std::cout << "     fFloat = " << fFloat[i] << std::endl;
            std::cout << "     fDouble = " << fDouble[i] << std::endl;
         }
     }

};

// _______________________________________________________________

class TJsonEx4 : public TJsonEx1 {
   protected:
      char        fStr1[100];
      const char* fStr2{nullptr};
      int         fDummy2{0};
      const char* fStr3{nullptr};
      const char* fStr4{nullptr};
   public:
      TJsonEx4()
      {
        memset(fStr1, 0, sizeof(fStr1));
      }

      virtual ~TJsonEx4()
      {
         delete[] fStr2;
         delete[] fStr3;
         delete[] fStr4;
      }

      void Init()
      {
         TJsonEx1::Init();
         strcpy(fStr1, "Value of string 1");
         fDummy2 = 1234567;
         fStr3 = new char[1000];
         strcpy((char*)fStr3, "***\t\n/t/n************** Long Value of string 3 *****************************************************************************************************************************************************************************************************************************************************************************************************************");
         fStr4 = new char[1000];
         strcpy((char*)fStr4, "--- normal string value ---");
      }

      void Print() override
      {
          TJsonEx1::Print();
          std::cout << "   fStr1 = " << fStr1 << std::endl;
          std::cout << "   fStr2 = " << (fStr2 ? fStr2 : "null") << std::endl;
          std::cout << "   fDummy2 = " << fDummy2 << std::endl;
          std::cout << "   fStr3 = " << fStr3 << std::endl;
      }
};

// _______________________________________________________________

class TJsonEx5 {
  protected:

     TJsonEx1    fObj1;
     TJsonEx2    fObj2;

     TJsonEx1    *fPtr1{nullptr};
     TJsonEx2    *fPtr2{nullptr};

     TJsonEx1    *fSafePtr1;   //->
     TJsonEx2    *fSafePtr2;   //->

   public:
     TJsonEx1    fObj3;
     TJsonEx1    *fPtr3{nullptr};
     TJsonEx1    *fSafePtr3;   //->

     TJsonEx5()
     {
        fSafePtr1 = new TJsonEx1;
        fSafePtr2 = new TJsonEx2;
        fSafePtr3 = new TJsonEx1;
     }

     void Init()
     {
       fObj1.Init();
       fObj2.Init();
       fObj3.Init();

       fPtr1 = new TJsonEx1;
       fPtr2 = new TJsonEx2;
       fPtr3 = fPtr1;

       fPtr1->Init();
       fPtr2->Init();

       fSafePtr1->Init();
       fSafePtr2->Init();
       fSafePtr3->Init();
     }

     virtual ~TJsonEx5()
     {
        delete fSafePtr1;
        delete fSafePtr2;
        delete fSafePtr3;
     }

     TJsonEx2&  GetObj2() { return fObj2; }
     TJsonEx2*  GetPtr2() { return fPtr2; }
     void      SetPtr2(TJsonEx2* ptr) { fPtr2 = ptr; }
     TJsonEx2*  GetSafePtr2() { return fSafePtr2; }


     void Print()
     {
        std::cout << std::endl << "!!!!!!!!!! fObj1 !!!!!!!!!!!" << std::endl;
        fObj1.Print();

        std::cout << std::endl << "!!!!!!!!!! fObj2 !!!!!!!!!!!" << std::endl;
        fObj2.Print();

        std::cout << std::endl << "!!!!!!!!!! fObj3 !!!!!!!!!!!" << std::endl;
        fObj3.Print();

        std::cout << std::endl << "!!!!!!!!!! fPtr1 !!!!!!!!!!!" << std::endl;
        if (fPtr1) fPtr1->Print();

        std::cout << std::endl << "!!!!!!!!!! fPtr2 !!!!!!!!!!!" << std::endl;
        if (fPtr2) fPtr2->Print();

        std::cout << std::endl << "!!!!!!!!!! fPtr3 !!!!!!!!!!!" << std::endl;
        if (fPtr3) fPtr3->Print();

        std::cout << std::endl << "!!!!!!!!!! fSafePtr1 !!!!!!!!!!!" << std::endl;
        if (fSafePtr1) fSafePtr1->Print();

        std::cout << std::endl << "!!!!!!!!!! fSafePtr2 !!!!!!!!!!!" << std::endl;
        if (fSafePtr2) fSafePtr2->Print();

        std::cout << std::endl << "!!!!!!!!!! fSafePtr3 !!!!!!!!!!!" << std::endl;
        if (fSafePtr3) fSafePtr3->Print();
     }
};


// _______________________________________________________________


class TJsonEx6 {
  protected:

     TJsonEx1    fObj1[3];
     TJsonEx2    fObj2[3];

     TJsonEx1*   fPtr1[3];
     TJsonEx2*   fPtr2[3];

     TJsonEx1*   fSafePtr1[3];   //->
     TJsonEx2*   fSafePtr2[3];   //->

   public:
     TJsonEx1    fObj3[3];
     TJsonEx1*   fPtr3[3];

     TJsonEx1*   fSafePtr3[3];   //->
     TJsonEx2*   fSafePtr33[3][3];   //->
     std::string  fStringArr234[2][3][4];

     TJsonEx2*   GetObj2() { return fObj2; }
     TJsonEx2**  GetPtr2() { return fPtr2; }
     TJsonEx2**  GetSafePtr2() { return fSafePtr2; }

     TJsonEx6()
     {
        for (int n=0;n<3;n++) {
           fPtr1[n] = nullptr;
           fPtr2[n] = nullptr;
           fPtr3[n] = nullptr;

           fSafePtr1[n] = new TJsonEx1();
           fSafePtr2[n] = new TJsonEx2();
           fSafePtr3[n] = new TJsonEx1();
           for (int k=0;k<3;k++)
              fSafePtr33[n][k] = new TJsonEx2();
         }
     }

     virtual ~TJsonEx6() { }

     void Init()
     {
        for (int n=0;n<3;n++) {
          fObj1[n].Init();
          fObj2[n].Init();
          fObj3[n].Init();

          fPtr1[n] = new TJsonEx1();
          fPtr2[n] = new TJsonEx2();
          fPtr3[n] = fPtr1[n];

          fPtr1[n]->Init();
          fPtr2[n]->Init();

           fSafePtr1[n]->Init();
           fSafePtr2[n]->Init();
           fSafePtr3[n]->Init();
           for (int k=0;k<3;k++)
              fSafePtr33[n][k]->Init();

        }

        for (int n=0;n<2;n++)
          for (int k=0;k<3;k++)
             for (int j=0;j<4;j++)
              fStringArr234[n][k][j] = Form("%d-%d-%d",n,k,j);
     }

     void Print()
     {
        for (int n=0;n<3;n++) {
           std::cout << std::endl << "!!!!!!!!!! fObj1["<<n<<"] !!!!!!!!!!!" << std::endl;
           fObj1[n].Print();

           std::cout << std::endl << "!!!!!!!!!! fObj2["<<n<<"] !!!!!!!!!!!" << std::endl;
           fObj2[n].Print();

           std::cout << std::endl << "!!!!!!!!!! fObj3["<<n<<"] !!!!!!!!!!!" << std::endl;
           fObj3[n].Print();

           std::cout << std::endl << "!!!!!!!!!! fPtr1["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fPtr1[n]) fPtr1[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fPtr2["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fPtr2[n]) fPtr2[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fPtr3["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fPtr3[n]) fPtr3[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fSafePtr1["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fSafePtr1[n]) fSafePtr1[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fSafePtr2["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fSafePtr2[n]) fSafePtr2[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fSafePtr3["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fSafePtr3[n]) fSafePtr3[n]->Print();
        }
     }
};

// _______________________________________________________________


class TJsonEx7 {
   public:
      std::string            fStr1;
      std::string            fStr2;
      std::string           *fStrPtr1{nullptr};
      std::string           *fStrPtr2{nullptr};

      std::string            fStrArr[3];
      std::string           *fStrPtrArr[3];

      bool                        fBoolArr[10];
      std::vector<double>         fVectDouble;
      std::vector<bool>           fVectBool;
      std::vector<double>         fVectDoubleArr[3];

      std::vector< double >      *fVectPtrDouble{nullptr};
      std::vector< double >      *fVectPtrDoubleArr[3];

      std::vector<TJsonEx1>        fVectEx1;
      std::vector<TJsonEx1*>       fVectEx1Ptr;
      std::vector<TJsonEx2>        fVectEx2;
      std::vector<TJsonEx3>        fVectEx3;

      std::vector<TBox>           fVectBox;

      std::vector<TNamed>         fVectNames;

      std::vector<TJsonEx11>      fVectEx11;

      int                         fABC{0};

      std::vector<std::string>    fVectString;
      std::vector<std::string*>   fVectStringPtr;

      std::list<double>           fListDouble;
      std::list<bool>             fListBool;

      std::list<TJsonEx1>         fListEx1;
      std::list<TJsonEx1*>        fListEx1Ptr;

      std::deque<double>          fDequeDouble;
      std::deque<bool>            fDequeBool;
      std::deque<TJsonEx1>        fDequeEx1;
      std::deque<TJsonEx1*>       fDequeEx1Ptr;

      std::map<int,double>        fMapIntDouble;
      std::map<TString,int>       fMapStrInt;
      std::map<int,TJsonEx1>      fMapIntEx1;
      std::map<int,TJsonEx1*>     fMapIntEx1Ptr;
      std::multimap<int,double>   fMultimapIntDouble;

      std::set<double>            fSetDouble;
      std::multiset<double>       fMultisetDouble;

      std::bitset<16>             fBitsSet16;
      std::bitset<64>             fBitsSet64;

   TJsonEx7()
   {

      for (int n=0;n<3;n++) {
        fStrPtrArr[n] = nullptr;
        fVectPtrDoubleArr[n] = nullptr;
      }

      for (int n=0;n<10;n++) fBoolArr[n] = false;

      for (int k=0;k<16;++k)
         fBitsSet16.set(k, false);

      for (int k=0;k<64;++k)
        fBitsSet64.set(k, false);
   }

   virtual ~TJsonEx7()
   {
   }

   void Init(int numelem = 5) {

      if (numelem <= 0) return;

      fABC = numelem;

      fStr1 = "String with special characters: \" & < >";
      fStr2 = "Very long Value of STL string// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8";

      fStrArr[0] = "Value of fStrArr[0]";
      fStrArr[1] = "Value of fStrArr[1]";
      fStrArr[2] = "Value of fStrArr[2]";

      fStrPtr1 = new std::string("Value of < >  &lt; &gt; string fStrPtr1");
      fStrPtr2 = nullptr; // new std::string("Value of string fStrPtr2");

      fStrPtrArr[0] = new std::string("value of fStrPtrArr[0]");
      fStrPtrArr[1] = new std::string("value of fStrPtrArr[1]");
      fStrPtrArr[2] = new std::string("value of fStrPtrArr[2]");

      for (int n=0;n<10;n++)
         fBoolArr[n] = (n%3 == 1);

      for (int n=0;n<numelem;n++) {

         fVectDouble.push_back(n*3);
         fVectBool.push_back(n%2 == 1);

         for (int k=0;k<3;++k)
            fVectDoubleArr[k].push_back(k*10+n);

         fVectEx1.push_back(TJsonEx1());
         fVectEx1.back().Init(n+1);
         fVectEx2.push_back(TJsonEx2());
         fVectEx2.back().Init(n+1);
         fVectEx3.push_back(TJsonEx3());
         fVectEx3.back().Init(n+3);

         fVectEx11.push_back(TJsonEx11());
         fVectEx11.back().Init(n+1);
      }

      fVectPtrDouble = new std::vector<double>;
      for (int n=0;n<numelem;n++)
         fVectPtrDouble->push_back(n*3);

      for (int i=0;i<3;i++) {
         fVectPtrDoubleArr[i] = new std::vector<double>;
         for (int n=0;n<numelem;n++)
            fVectPtrDoubleArr[i]->push_back(i*100 + n*3);
      }

      int sz3 = numelem>=3 ? numelem/3 : numelem;

      for (int n=0;n<numelem;n++) {
         TBox box(n*11,n*22,n*33,n*44);
         fVectBox.push_back(box);

         TNamed name(Form("Name%d",n),Form("Title%d",n));
         fVectNames.push_back(name);
      }


      for (int n=0;n<sz3;n++) {
        TJsonEx1 *ex1 = new TJsonEx1;
        ex1->Init(n*2+3);
        fVectEx1Ptr.push_back(ex1);
        fVectEx1Ptr.push_back(ex1);
        fVectEx1Ptr.push_back(ex1);
      }



      int sz5 = numelem>=2 ? numelem/2 : 1;

      for (int n=0;n<sz5;n++) {
        fVectString.push_back(Form("string %d content",n));
        fVectStringPtr.push_back(new std::string(Form("string pointer %d content",n)));
      }


      for (int n=0;n<numelem;n++) {

         fListDouble.push_back(n*4);
         fListBool.push_back(n%2 == 1);

         fListEx1.push_back(TJsonEx1());
         fListEx1.back().Init(n);
         if (n == 1) fListEx1Ptr.push_back(fListEx1Ptr.front()); else
         if (n%3 == 2) fListEx1Ptr.push_back(nullptr); else {
            fListEx1Ptr.push_back(new TJsonEx1);
            fListEx1Ptr.back()->Init(n+7);
         }

         fDequeDouble.push_back(n*4);
         fDequeBool.push_back(n%2 == 1);
         fDequeEx1.push_back(TJsonEx1());
         fDequeEx1.back().Init(n+11);

         if (n == numelem-2) fDequeEx1Ptr.push_back(fDequeEx1Ptr.front()); else
         if (n%3 == 1) fDequeEx1Ptr.push_back(nullptr); else {
            fDequeEx1Ptr.push_back(new TJsonEx1);
            fDequeEx1Ptr.back()->Init(n+1);
         }

         fMapIntDouble[n] = n*5;
         fMapStrInt[Form("Str%d",n)] = n + 10;
         fMapIntEx1[n] = TJsonEx1(); fMapIntEx1[n].Init(n+8);
         fMapIntEx1Ptr[n] = new TJsonEx1; fMapIntEx1Ptr[n]->Init(n+3);

         fMultimapIntDouble.insert(std::pair<int,double>(n,n*6));
         fMultimapIntDouble.insert(std::pair<int,double>(n,1000+n*6));
         fSetDouble.insert(n+2);
         fMultisetDouble.insert((n+1)*17);
      }

      for (int k=0;k<16;++k)
         fBitsSet16.set(k, k%2);

      for (int k=0;k<64;++k)
        fBitsSet64.set(k, k%3);
   }



   void Print()
   {
      std::cout << "Do not print everything, just something..." << std::endl;

      std::cout << " fStr1 = " << fStr1 << std::endl;

      std::cout << " fStrArr[1] = " << fStrArr[1] << std::endl;

      //std::cout << " fVectEx1.back().Print()" << std::endl;
      //fVectEx1.back().Print();
      //std::cout << " fVectEx1Ptr.size() = " << fVectEx1Ptr.size() << std::endl;
      //std::cout << " fVectStringPtr.size() = " << fVectStringPtr.size() << std::endl;
      //std::cout << " fMapIntEx1Ptr.size() = " << fMapIntEx1Ptr.size() << std::endl;
   }
};

// _______________________________________________________________

class TJsonEx8 : public std::vector<int> {
   public:
     int                    fInt{0};
     std::string            fStdString;

   TJsonEx8() = default;

   virtual ~TJsonEx8() {}

   void Init(int numelem = 11)
   {
     for (int n=0;n<numelem;n++)
       push_back((n+2)*23);
     fInt = 12345;
     fStdString = "Value of STL string, numelem = ";
     fStdString += std::to_string(numelem);
   }

   void Print()
   {
      std::cout << "vector size = " << size() << std::endl;
      for (unsigned n=0;n<size(); n++)
        std::cout << "vector.at(" << n << ") = " << at(n) << std::endl;
      std::cout << " fInt = " << fInt << std::endl;
      std::cout << " fStdString = " << fStdString << std::endl;
   }
};

//__________________________________________________________________

class TJsonEx9 {
   public:
      int fCnt{0};
      TString* fStr{nullptr};     //[fCnt]
      TString* fStr2[3]; //[fCnt]
//      TString** fStr3;   //[fCnt]
      TString* fStr4[2][5]; //[fCnt]
      TString** fStr6[3]; //[fCnt]
      TNamed* fNm2[3]; //[fCnt]
      TNamed* fNm4[2][5]; //[fCnt]
      TNamed** fNm6[3]; //[fCnt]

      TArrayI* fArr2[3]; //[fCnt]
      TArrayI* fArr4[2][5]; //[fCnt]
      TArrayI** fArr6[3]; //[fCnt]

   TJsonEx9()
   {
      for (int k=0;k<3;++k) {
         fStr2[k] = nullptr;
         fNm2[k] = nullptr;
         fStr6[k] = nullptr;
         fNm6[k] = nullptr;
         fArr2[k] = nullptr;
         fArr6[k] = nullptr;
      }
      for (int k1=0;k1<2;++k1)
         for (int k2=0;k2<5;++k2) {
            fStr4[k1][k2] = nullptr;
            fNm4[k1][k2] = nullptr;
            fArr4[k1][k2] = nullptr;
         }
//      fStr3 = 0;
   }

   void Init(int cnt = 3)
   {
      fCnt = cnt;

      fStr = new TString[fCnt];
      for (int n=0;n<fCnt;++n) {
         fStr[n].Form("String%d",n);
      }

      for (int k=0;k<3;++k) {
         fStr2[k] = new TString[fCnt];
         for (int n=0;n<fCnt;++n)
            fStr2[k][n].Form("String[%d][%d]",k,n);
         fNm2[k] = new TNamed[fCnt];
         for (int n=0;n<fCnt;++n) {
            fNm2[k][n].SetName(Form("Name[%d][%d]",k,n));
            fNm2[k][n].SetTitle(Form("Title[%d][%d]",k,n));
         }
         fArr2[k] = new TArrayI[fCnt];
         for (int n=0;n<fCnt;++n) {
            fArr2[k][n].Set(5);
            fArr2[k][n].Reset(n);
         }

         fStr6[k] = new TString*[fCnt];
         for (int n=0;n<fCnt;++n) {
            fStr6[k][n] = nullptr;
            if ((k+n) % 3 == 0) continue;

            fStr6[k][n] = new TString;
            fStr6[k][n]->Form("String[%d][%d]",k,n);
         }

         fNm6[k] = new TNamed*[fCnt];
         for (int n=0;n<fCnt;++n) {
            fNm6[k][n] = nullptr;
            if ((k+n) % 3 == 0) continue;

            fNm6[k][n] = new TNamed;
            fNm6[k][n]->SetName(Form("Name[%d][%d]",k,n));
            fNm6[k][n]->SetTitle(Form("Title[%d][%d]",k,n));
         }

         fArr6[k] = new TArrayI*[fCnt];
         for (int n=0;n<fCnt;++n) {
            fArr6[k][n] = nullptr;
            if ((k+n) % 3 == 0) continue;

            fArr6[k][n] = new TArrayI(5);
            fArr6[k][n]->Reset(n);
         }

      }

      for (int k1=0;k1<2;++k1)
         for (int k2=0;k2<5;++k2) {
            fStr4[k1][k2] = new TString[fCnt];
            fNm4[k1][k2] = new TNamed[fCnt];
            fArr4[k1][k2] = new TArrayI[fCnt];
            for (int n=0;n<fCnt;++n) {
               fStr4[k1][k2][n].Form("String[%d][%d][%d]",k1,k2,n);
               fNm4[k1][k2][n].SetName(Form("Name[%d][%d][%d]",k1,k2,n));
               fNm4[k1][k2][n].SetTitle(Form("Name[%d][%d][%d]",k1,k2,n));
               fArr4[k1][k2][n].Set(5);
               fArr4[k1][k2][n].Reset(n);
            }
         }
   }

   virtual ~TJsonEx9() {
   }
};

// _________________________________________________________________________

class TJsonEx10 {
   public:
      TString fStr0;
      TNamed  fName0;
      TObject fObj0;

      TString fStr[10];
      TNamed  fName[10];
      TObject fObj[10];

      TJsonEx10() {}
      virtual ~TJsonEx10() {}

      void Init() {
         fStr0 = "SimpleString";
         fName0.SetName("SimpleName");
         fName0.SetTitle("SimpleTitle");
         fObj0.SetUniqueID(10000);

         for (int n=0;n<10;++n) {
            fStr[n].Form("String%d",n);
            fName[n].SetName(Form("Name%d",n));
            fName[n].SetTitle(Form("Title%d",n));
            fObj[n].SetUniqueID(n+100);
         }
      }
};

// ______________________________________________________________________________________

class TJsonEx12  {
public:
   std::vector<TJsonEx1> vect1;
   std::vector<TJsonEx9> vect9;

   TJsonEx12() = default;
   virtual ~TJsonEx12() {}

    void Init(int cnt = 7)
    {
      vect1.resize(cnt);
      vect9.resize(cnt);
      for (int n=0;n<cnt;++n) {
         vect1[n].Init(n+1);
         vect9[n].Init((n % 3) + 1);
      }
   }

};

// ______________________________________________________________________________________

class TJsonEx13  {
public:
   std::set<TJsonEx1> set1;
   std::map<TJsonEx1, std::set<TJsonEx1>> map1;
   std::map<int, TNamed> map2;
   std::vector<TNamed> vect1;
   std::vector<std::vector<std::vector<std::vector<int>>>> vvvv;
   std::map<std::string, int> map_obj;  /// JSON_object

   TJsonEx13() = default;
   virtual ~TJsonEx13() {}

   void Init(int cnt = 4)
   {
      std::vector<int> v;
      std::vector<std::vector<int>> vv;
      std::vector<std::vector<std::vector<int>>> vvv;
      TJsonEx1 ex1;
      for (int n=0;n<cnt;++n) {
         TNamed named(Form("name%d",n), Form("title%d",n));
         ex1.Init(n);
         set1.insert(ex1);
         map1[ex1] = set1;
         map2[n] = named;
         vect1.push_back(named);
         v.emplace_back(n);

         map_obj[std::string("name") + std::to_string(n)] = n*11;
      }

      for (int n=0;n<cnt;++n)
         vv.emplace_back(v);
      if (cnt>1) cnt--;

      for (int n=0;n<cnt;++n)
         vvv.emplace_back(vv);

      if (cnt>1) cnt--;
      for (int n=0;n<cnt;++n)
         vvvv.emplace_back(vvv);
   }
};

// ______________________________________________________________________________________

class TJsonEx14 {
public:

   TObjArray fObjArray;
   TList fList;
   TClonesArray fClones;
   TMap fMap;

   TJsonEx14()
   {
      fObjArray.SetOwner(kTRUE);
      fList.SetOwner(kTRUE);
      fMap.SetOwner(kTRUE);
   }

   void Init(int cnt = 7)
   {
      for(int n=0;n<cnt;n++) {
         TNamed* nn = new TNamed(Form("ObjArr%d",n), Form("ObjArrTitle%d",n));
         fObjArray.Add(nn);
      }

      for(int n=0;n<cnt;n++) {
         TBox* b = new TBox(n*10,n*100,n*20,n*200);
         fList.Add(b, Form("option_%d_option",n));
      }

      fClones.SetClass("TBox", cnt);
      for(int n=0;n<cnt;n++)
         new (fClones[n]) TBox(n*7,n*77,n*17,n*27);

     for (int n=0;n<cnt;n++) {
         TObjString* str = new TObjString(Form("Str%d",n));
         TNamed* nnn = new TNamed(Form("Name%d",n), Form("Title%d",n));
         fMap.Add(str,nnn);
      }

   }
};

// ______________________________________________________________________________________

bool testJsonReading(TString &json)
{
   TClass *cl = nullptr;

   void *obj = TBufferJSON::ConvertFromJSONAny(json, &cl);

   if (!obj) {
      printf("Fail to read object from json %s...\n", json(0,30).Data());
      return false;
   }

   if (!cl) {
      printf("Fail to get class from json %s ...\n", json(0,30).Data());
      return false;
   }

   TString json2 = TBufferJSON::ConvertToJSON(obj, cl);

   bool res = (json == json2);

   printf("%s store/read/store %s len1:%d len2:%d\n", cl->GetName(), res ? "MATCHED" : "FAILED", json.Length(), json2.Length());

   if (!res) {
      Int_t errcnt = 0, minlen = json.Length() < json2.Length() ? json.Length() : json2.Length();
      for (Int_t p = 0; p < minlen-30; p+=30) {
         TString part1 = json(p,30), part2 = json2(p,30);
         if (part1 != part2) {
            printf("DIFF at pos:%d\n%s\n%s\n", p, part1.Data(), part2.Data());
            if (++errcnt > 5) break;
         }
      }
   }

   cl->Destructor(obj);

   return res;
}


#endif
