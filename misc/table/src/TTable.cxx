// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   03/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTable                                                                 //
//                                                                        //
// Wraps the array of the plain C-structures (one C-structure per element)//
//                                                                        //
// class TTable provides the automatic schema evolution for               //
// the derived "table" classes saved with ROOT format.                    //
//                                                                        //
// "Automatic Schema evolution" provides:                                 //
//   -  skipping data-member if it is not present for the current         //
//      implementation of the "table" but was present at the time the     //
//      table was written;                                                //
//   -  assign a default value ZERO for the brand-new data-members,       //
//      those were not in the structure when the object was written but   //
//      present now;                                                      //
//   -  trace propely any change in the order of the data-members         //
//                                                                        //
// To enjoy this class one has to derive his/her own custom class:        //
//                                                                        //
// St_dst_track_Table.h:                                                  //
// ---------------------                                                  //
//  #ifndef STAF_St_dst_track_Table                                       //
//  #define STAF_St_dst_track_Table                                       //
//                                                                        //
//  #include "TTable.h"                                                   //
//                                                                        //
// // #include "dst_track.h"  the C-structure defintion may be kept       //
//                            separately                                  //
//  typedef struct dst_track_st {                                         //
//     float r0;             /* radius at start (cm). See also comments*/
//     float phi0;           /* azimuthal angle at start (deg)         */
//     float z0;             /* z-coord. at start (cm)                 */
//     float psi;            /* azimuthal angle of pT vector (deg)     */
//     float tanl;           /* tan(dip) =pz/pt at start               */
//     float invpt;          /* 1/pt at start (GeV/c)^(-1)             */
//     float curvature;      /* Track curvature (1/cm)                 */
//     float covar[15];      /* full covariance matrix                 */
//     float chisq[2];       /* Chi-square per degree of freedom       */
//     float x_first[3];     /* coord. of first measured point (cm)    */
//     float x_last[3];      /* coord. of last measured point (cm)     */
//     float length;         /* from first to last point (cm)          */
//     float impact;         /* primary vertex (cm)                    */
//     unsigned long map[2]; /* extrap. info. (see preceeding comments)*/
//     int id;               /* Primary key (see comments)             */
//     int iflag;            /* bitmask quality info. (see comments)   */
//     int det_id;           /* Detector id information                */
//     int method;           /* Track finding/fitting method, packed   */
//     int pid;              /* Geant particle ID for assumed mass     */
//     int n_point;          /* SVT, TPC, FTPC component #s are packed */
//     int n_max_point;      /* SVT, TPC, FTPC component #s are packed */
//     int n_fit_point;      /* SVT, TPC, FTPC component #s are packed */
//     int icharge;          /* Particle charge in units of |e|        */
//     int id_start_vertex;  /* final fit and primary track candidates */
//  } DST_TRACK_ST;                                                       //
//                                                                        //
//  class St_dst_track : public TTable                                    //
//  {                                                                     //
//   public:                                                              //
//     ClassDefTable(St_dst_track,dst_track_st)                           //
//     ClassDef(St_dst_track,2) //C++ wrapper for <dst_track> StAF table  //
//  };                                                                    //
//  #endif                                                                //
// ---------------------                                                  //
//                                                                        //
//  where the CPP macro defines several convinient methods for the        //
//  "table" class (see: $ROOTSYS/table/inc/Ttypes.h for details:          //
//                                                                        //
//  #define ClassDefTable(className,structName)
//    protected:
//       static TTableDescriptor *fgColDescriptors;
//       virtual TTableDescriptor *GetDescriptorPointer() const { return fgColDescriptors;}
//       virtual void SetDescriptorPointer(TTableDescriptor *list) const { fgColDescriptors = list;}
//    public:
//      typedef structName* iterator;
//      className() : TTable(_QUOTE_(className),sizeof(structName))    {SetType(_QUOTE_(structName));}
//      className(const char *name) : TTable(name,sizeof(structName)) {SetType(_QUOTE_(structName));}
//      className(Int_t n) : TTable(_QUOTE_(className),n,sizeof(structName)) {SetType(_QUOTE_(structName));}
//      className(const char *name,Int_t n) : TTable(name,n,sizeof(structName)) {SetType(_QUOTE_(structName));}
//      structName *GetTable(Int_t i=0) const { return ((structName *)GetArray())+i;}
//      structName &operator[](Int_t i){ assert(i>=0 && i < GetNRows()); return *GetTable(i); }
//      const structName &operator[](Int_t i) const { assert(i>=0 && i < GetNRows()); return *((const structName *)(GetTable(i))); }
//      structName *begin() const  {                      return GetNRows()? GetTable(0):0;}
//      structName *end()   const  {Int_t i = GetNRows(); return          i? GetTable(i):0;}
//                                                                        //
//  The class implementation file may 2 lines and look as follows:        //
//  (for the example above):                                              //
//                                                                        //
//  St_dst_track_Table.cxx:                                               //
//  -----------------------                                               //
//       #include "St_dst_track_Table.h"                                  //
//       TableClassImpl(St_dst_track, dst_track_st)                       //
//  -----------------------                                               //
//  LinkDef.h                                                             //
//  -----------------------                                               //
//  To provide ROOT I/O for this class TWO CINT dictonary entries         //
//  should be defined with your custom LinkDef.h file                     //
//     1. First entry (as usually) for the class derived from TTable      //
//        for example:                                                    //
// #pragma C++ class St_dst_track                                         //
//     2. Second entry for the C-structure wrapped into the class.        //
//         Since C-structuire is not derived from TObject it must be      //
//         properly defined as "foreign" ROOT class                       //
//    #pragma C++ class dst_track_st+;                                    //
//  -----------------------                                               //
// meta-variables i$ and n$ introduced                                    //
// where "i$" stands for the current row index                            //
//       "n$" stands for the total number of rows                         //
// meta-variable can be used along the normal                             //
// table column names in the expressions (see for example                 //
// method TTable::Draw                                                    //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include <assert.h>

#ifdef WIN32
# include <float.h>
#endif

//#if ROOT_VERSION_CODE >= ROOT_VERSION(3,03,5)
#include "Riosfwd.h"
#include "Riostream.h"
//#include <iomanip.h>

// #endif

#include "TROOT.h"
#include "TBaseClass.h"
#include "TSystem.h"
#include "TBuffer.h"
#include "TMath.h"
#include "TClass.h"
#include "TBrowser.h"
#include "TString.h"
#include "Api.h"
#include "TDataSetIter.h"
#include "TTable.h"
#include "TTableDescriptor.h"
#include "TColumnView.h"

#include "TGaxis.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TVirtualPad.h"
#include "TEventList.h"
#include "TPolyMarker.h"
#include "TView.h"
#include "TGaxis.h"
#include "TPolyMarker3D.h"

#include "THLimitsFinder.h"

#include "TTableMap.h"

static TH1 *gCurrentTableHist = 0;

static const char *gDtorName = "dtor";
static   Int_t         gNbins[4] = {100,100,100,100};     //Number of bins per dimension
static   Float_t       gVmin[4]  = {0,0,0,0};             //Minima of varexp columns
static   Float_t       gVmax[4]  = {20,20,20,20};         //Maxima of varexp columns

const char *TTable::fgTypeName[] = {
   "NAN", "float", "int", "long", "short", "double",
   "unsigned int", "unsigned long","unsigned short",
   "unsigned char", "char", "Ptr_t"
};

//______________________________________________________________________________
static void ArrayLayout(UInt_t *layout,const UInt_t *size, Int_t dim)
{
  //
  // ArrayLayout - calculates the array layout recursively
  //
  // Input:
  // -----
  // dim   - dimension of the targeted array
  // size  - the max index for each dimension
  //
  // Output:
  // ------
  // layout - the "start index" for each dimension of an array
  //

   if (dim && layout && size) {
      if (++layout[dim-1] >= size[dim-1]) {
         layout[dim-1] = 0;
         dim--;
         ArrayLayout(layout,size, dim);
      }
   }
}

ClassImp(TTable)

//______________________________________________________________________________
TTableDescriptor *TTable::GetTableDescriptors() const {
 // protected: create a new TTableDescriptor descriptor for this table
   assert(0);
   return new TTableDescriptor(this);
}

//______________________________________________________________________________
void TTable::AsString(void *buf, EColumnType type, Int_t width,ostream &out) const
{
  //
  // AsString represents the value provided via "void *b" with type defined
  //          by "name"
  //
  //   void *buf  - the pointer to the value to be printed out.
  //        type  - the basic data type for the value above
  //       width  - the number of psotion to be used to print the value out
  //
   int prevPrec = out.precision();
   const std::ios_base::fmtflags prevFmt = out.flags();
   
   switch (type) {
      case kFloat:
         out << dec  << setw(width) << setprecision(width-3) << *(float *)buf;
         break;
      case kInt:
         out << dec  <<  setw(width) << *(int *)buf;
         break;
      case kLong:
         out << dec  << setw(width) << *(long *)buf;
         break;
      case kShort:
         out << dec  << setw(width) << *(short *)buf;
         break;
      case kDouble:
         out << dec  << setw(width) << setprecision(width-3) << *(double *)buf;
         break;
      case kUInt:
         out << dec  << setw(width) << *(unsigned int *)buf;
         break;
      case kULong:
         out << dec  << setw(width) << *(unsigned long *)buf;
         break;
      case kUShort:
         out  << setw(width) << "0x" << hex << *(unsigned short *)buf;
         break;
      case kUChar:
         out  << setw(width) << "0x" << hex << int(*(unsigned char *)buf);
         break;
      case kChar:
         out << setw(width) << *(char *)buf;
         break;
      case kBool:
         out << setw(width) << *(bool *)buf;
         break;
      case kPtr:
         out << "->" << setw(width) << *(void **)buf;
         break;
      default:
         out << "\"NaN\"";
         break;
   };
   out.precision(prevPrec);
   out.setf(prevFmt);
}

//________________________________________________________________________
const char *TTable::GetTypeName(TTable::EColumnType type)
{  
   //return table type name
   return  fgTypeName[type]; 
}

//________________________________________________________________________
TTable::EColumnType TTable::GetTypeId(const char *typeName)
{
   // return the Id of the C basic type by given name
   // return kNAN if the name provided fits no knwn basic name.
   //
   Int_t allTypes = sizeof(fgTypeName)/sizeof(const char *);
   for (int i = 0; i < allTypes; i++)
   if (!strcmp(fgTypeName[i],typeName)) return EColumnType(i);
   return kNAN;
}

//______________________________________________________________________________
const void *TTable::At(Int_t i) const
{
   // Returns a pointer to the i-th row of the table
   if (!BoundsOk("TTable::At", i)) {
       Warning("TTable::At","%s.%s",GetName(),GetType());
      i = 0;
   }
   return (const void *)(fTable+i*fSize);
}

//______________________________________________________________________________
Int_t TTable::CopyRows(const TTable *srcTable, Long_t srcRow, Long_t dstRow, Long_t nRows, Bool_t expand)
{
 // CopyRows copies nRows from starting from the srcRow of srcTable
 // to the dstRow in this table upto nRows or by the end of this table.
 //
 // This table if automaticaly increased if expand = kTRUE.
 // The old values of this table rows are to be destroyed and
 // replaced with the new ones.
 //
 // PARAMETERS:
 //   srcTable - a pointer to the table "donor"
 //   srcRow   - the index of the first row of the table donor to copy from
 //   dstRow   - the index of the first row of this table to copy to
 //   nRows    - the total number of rows to be copied. This table will be expanded
 //              as needed if expand = kTRUE (it is kFALSE "by default")
 //          = 0 to copy ALL remain rows from the srcTable.
 //   expand   - flag whether this table should reallocated if needed.
 //
 // RETURN:
 //          the number of the rows been copied

   assert(!TestBit(kIsNotOwn));
   if (!(srcTable && srcTable->GetNRows()) || srcRow > srcTable->GetNRows()-1   )   return 0;
   if (strcmp(GetType(),srcTable->GetType())) {
      // check this table current capacity
      if (!nRows) nRows = srcTable->GetNRows();
      Long_t tSize = GetTableSize();
      Long_t extraRows = (tSize - dstRow) - nRows;
      if (extraRows < 0) {
         if (expand) {
            ReAllocate(tSize - extraRows);
            extraRows = 0;
         }
         nRows += extraRows;
      }
      if (dstRow+nRows > GetNRows()) SetNRows(dstRow+nRows);
      ::memmove((*this)[dstRow],(*srcTable)[srcRow],(size_t)GetRowSize()*nRows);
      return nRows;
   } else
      Error("CopyRows",
           "This table is <%s> but the src table has a wrong type <%s>",GetType()
           ,srcTable->GetType());
   return 0;
}
//______________________________________________________________________________
void TTable::DeleteRows(Long_t indx, UInt_t nRows)
{
  // Delete one or several rows from the table
  //
  //  Int_t indx  - index of the first row to be deleted
  //  Int_t nRows - the total number of rows to be deleted
  //              = 1 "by default
   if (CopyRows(this, indx+nRows, indx, GetNRows()-indx-nRows))
      SetUsedRows(GetNRows() - nRows);
}
//______________________________________________________________________________
TH1  *TTable::Draw(TCut varexp, TCut selection, Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*-*-*Draw expression varexp for specified entries-*-*-*-*-*
//*-*                  ===========================================
//
//   This function accepts TCut objects as arguments.
//   Useful to use the string operator +
//         example:
//            table.Draw("x",cut1+cut2+cut3);
//
//   TCutG object with "CUTG" name can be created via the graphics editor.
//

   return TTable::Draw(varexp.GetTitle(), selection.GetTitle(), option, nentries, firstentry);
}

//______________________________________________________________________________
TH1 *TTable::Draw(const char *varexp00, const char *selection, Option_t *option,Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*-*-*Draw expression varexp for specified entries-*-*-*-*-*
//*-*                  ===========================================
//
//  varexp is an expression of the general form e1:e2:e3
//    where e1,etc is a C++ expression referencing a combination of the TTable columns
//          One can use two extra meta variable "i$" and "n$" along with the table
//          column names.
//          i$ is to involve the current row number
//          n$ refers the total num,ber of rows of this table provided by TTable::GetNRows()
//
//  Example:
//     varexp = x     simplest case: draw a 1-Dim distribution of column named x
//            = sqrt(x)            : draw distribution of sqrt(x)
//            = x*y/z
//            = y:sqrt(x) 2-Dim dsitribution of y versus sqrt(x)
//            = i$:sqrt(x) 2-Dim dsitribution of i versus sqrt(x[i])
//            = phep[0]:sqrt(phep[3]) 2-Dim dsitribution of phep[0] versus sqrt(phep[3])
//
//  Note that the variables e1, e2 or e3 may contain a boolean expression as well.
//  example, if e1= x*(y<0), the value histogrammed will be x if y<0
//  and will be 0 otherwise.
//
//  selection is a C++ expression with a combination of the columns.
//  The value corresponding to the selection expression is used as a weight
//  to fill the histogram.
//  If the expression includes only boolean operations, the result
//  is 0 or 1. If the result is 0, the histogram is not filled.
//  In general, the expression may be of the form:
//
//      value*(boolean expression)
//
//  if boolean expression is true, the histogram is filled with
//  a weight = value.
//  Examples:
//      selection1 = "x<y && sqrt(z)>3.2 && 6 < i$ && i$ < n$"
//      selection2 = "(x+y)*(sqrt(z)>3.2"
//      selection3 = "signal*(log(signal)>1.2)"
//  selection1 returns a weigth = 0 or 1
//  selection2 returns a weight = x+y if sqrt(z)>3.2
//             returns a weight = 0 otherwise.
//  selection3 returns a weight = signal if log(signal)>1.2
//
//  option is the drawing option
//      see TH1::Draw for the list of all drawing options.
//      If option contains the string "goff", no graphics is generated.
//
//  nentries is the number of entries to process (default is all)
//  first is the first entry to process (default is 0)
//
//     Saving the result of Draw to an histogram
//     =========================================
//  By default the temporary histogram created is called htemp.
//  If varexp0 contains >>hnew (following the variable(s) name(s),
//  the new histogram created is called hnew and it is kept in the current
//  directory.
//  Example:
//    tree.Draw("sqrt(x)>>hsqrt","y>0")
//    will draw sqrt(x) and save the histogram as "hsqrt" in the current
//    directory.
//
//  By default, the specified histogram is reset.
//  To continue to append data to an existing histogram, use "+" in front
//  of the histogram name;
//    table.Draw("sqrt(x)>>+hsqrt","y>0")
//      will not reset hsqrt, but will continue filling.
//
//     Making a Profile histogram
//     ==========================
//  In case of a 2-Dim expression, one can generate a TProfile histogram
//  instead of a TH2F histogram by specyfying option=prof or option=profs.
//  The option=prof is automatically selected in case of y:x>>pf
//  where pf is an existing TProfile histogram.
//
//     Saving the result of Draw to a TEventList
//     =========================================
//  TTable::Draw can be used to fill a TEventList object (list of entry numbers)
//  instead of histogramming one variable.
//  If varexp0 has the form >>elist , a TEventList object named "elist"
//  is created in the current directory. elist will contain the list
//  of entry numbers satisfying the current selection.
//  Example:
//    tree.Draw(">>yplus","y>0")
//    will create a TEventList object named "yplus" in the current directory.
//    In an interactive session, one can type (after TTable::Draw)
//       yplus.Print("all")
//    to print the list of entry numbers in the list.
//
//  By default, the specified entry list is reset.
//  To continue to append data to an existing list, use "+" in front
//  of the list name;
//    table.Draw(">>+yplus","y>0")
//      will not reset yplus, but will enter the selected entries at the end
//      of the existing list.
//

   if (GetNRows() == 0 || varexp00 == 0 || varexp00[0]==0) return 0;
   TString  opt;
//   char *hdefault = (char *)"htemp";
   const char *hdefault = "htemp";
   Int_t i,j,action;
   Int_t hkeep = 0;
   opt = option;
   opt.ToLower();
   char *varexp0 = StrDup(varexp00);
   char *hname = strstr(varexp0,">>");
   TH1 *oldh1 = 0;
   TEventList *elist = 0;
   Bool_t profile = kFALSE;

   gCurrentTableHist = 0;
   if (hname) {
      *hname  = 0;
      hname += 2;
      hkeep  = 1;
      i = strcspn(varexp0,">>");
      Bool_t hnameplus = kFALSE;
      while (*hname == ' ') hname++;
      if (*hname == '+') {
         hnameplus = kTRUE;
         hname++;
         while (*hname == ' ') hname++;
         j = strlen(hname)-1;
         while (j) {
            if (hname[j] != ' ') break;
            hname[j] = 0;
            j--;
         }
      }
      if (i) {
         oldh1 = (TH1*)gDirectory->Get(hname);
         if (oldh1 && !hnameplus) oldh1->Reset();
      } else {
         elist = (TEventList*)gDirectory->Get(hname);
         if (!elist) {
            elist = new TEventList(hname,selection,1000,0);
         }
         if (elist && !hnameplus) elist->Reset();
      }
   }
   if (!hname || *hname==0) {
      hkeep  = 0;
      if (gDirectory) {
         oldh1 = (TH1*)gDirectory->Get(hdefault);
         if (oldh1 ) { oldh1->Delete(); oldh1 = 0;}
      }
   }

   // Look for colons
   const Char_t *expressions[] ={varexp0,0,0,0,selection};
   Int_t maxExpressions = sizeof(expressions)/sizeof(Char_t *);
   Char_t *nextColon    = varexp0;
   Int_t colIndex       = 1;
   while ((nextColon = strchr(nextColon,':')) && ( colIndex < maxExpressions - 1 ) ) {
      *nextColon = 0;
      nextColon++;
      expressions[colIndex] = nextColon;
      colIndex++;
   }

   expressions[colIndex] = selection;


//--------------------------------------------------
   Printf(" Draw %s for <%s>\n", varexp00, selection);
   Char_t *exprFileName = MakeExpression(expressions,colIndex+1);
   if (!exprFileName) {
      delete [] varexp0;
      return 0;
   }

//--------------------------------------------------
//   if (!fVar1 && !elist) return 0;

//*-*- In case oldh1 exists, check dimensionality
   Int_t dimension = colIndex;

   TString title = expressions[0];
   for (i=1;i<colIndex;i++) {
      title += ":";
      title += expressions[i];
   }
   Int_t nsel = strlen(selection);
   if (nsel > 1) {
      if (nsel < 80-title.Length()) {
         title += "{";
         title += selection;
         title += "}";
      } else
         title += "{...}";
   }

   const Char_t *htitle = title.Data();

   if (oldh1) {
      Int_t mustdelete = 0;
      if (oldh1->InheritsFrom(TProfile::Class())) profile = kTRUE;
      if (opt.Contains("prof")) {
         if (!profile) mustdelete = 1;
      } else {
         if (oldh1->GetDimension() != dimension) mustdelete = 1;
      }
      if (mustdelete) {
         Warning("Draw","Deleting old histogram with different dimensions");
         delete oldh1; oldh1 = 0;
      }
   }
//*-*- Create a default canvas if none exists
   if (!gPad && !opt.Contains("goff") && dimension > 0) {
      gROOT->MakeDefCanvas();
   }
//*-*- 1-D distribution
   if (dimension == 1) {
      action = 1;
      if (!oldh1) {
         gNbins[0] = 100;
         if (gPad && opt.Contains("same")) {
            TH1 *oldhtemp = (TH1*)gPad->FindObject(hdefault);
            if (oldhtemp) {
               gNbins[0] = oldhtemp->GetXaxis()->GetNbins();
               gVmin[0]  = oldhtemp->GetXaxis()->GetXmin();
               gVmax[0]  = oldhtemp->GetXaxis()->GetXmax();
            } else {
               gVmin[0]  = gPad->GetUxmin();
               gVmax[0]  = gPad->GetUxmax();
            }
         } else {
            action = -1;
         }
      }
      TH1F *h1;
      if (oldh1) {
         h1 = (TH1F*)oldh1;
         gNbins[0] = h1->GetXaxis()->GetNbins();  // for proofserv
      } else {
         h1 = new TH1F(hname,htitle,gNbins[0],gVmin[0],gVmax[0]);
         if (!hkeep) {
            h1->SetBit(kCanDelete);
            h1->SetDirectory(0);
         }
         if (opt.Length() && opt[0] == 'e') h1->Sumw2();
      }

      EntryLoop(exprFileName,action, h1, nentries, firstentry, option);

//    if (!fDraw && !opt.Contains("goff")) h1->Draw(option);
      if (!opt.Contains("goff")) h1->Draw(option);

//*-*- 2-D distribution
   } else if (dimension == 2) {
      action = 2;
      if (!opt.Contains("same") && gPad)  gPad->Clear();
      if (!oldh1 || !opt.Contains("same")) {
         gNbins[0] = 40;
         gNbins[1] = 40;
         if (opt.Contains("prof")) gNbins[1] = 100;
         if (opt.Contains("same")) {
            TH1 *oldhtemp = (TH1*)gPad->FindObject(hdefault);
            if (oldhtemp) {
               gNbins[1] = oldhtemp->GetXaxis()->GetNbins();
               gVmin[1]  = oldhtemp->GetXaxis()->GetXmin();
               gVmax[1]  = oldhtemp->GetXaxis()->GetXmax();
               gNbins[0] = oldhtemp->GetYaxis()->GetNbins();
               gVmin[0]  = oldhtemp->GetYaxis()->GetXmin();
               gVmax[0]  = oldhtemp->GetYaxis()->GetXmax();
            } else {
               gNbins[1] = 40;
               gVmin[1]  = gPad->GetUxmin();
               gVmax[1]  = gPad->GetUxmax();
               gNbins[0] = 40;
               gVmin[0]  = gPad->GetUymin();
               gVmax[0]  = gPad->GetUymax();
            }
         } else {
            action = -2;
         }
      }
      if (profile || opt.Contains("prof")) {
         TProfile *hp;
         if (oldh1) {
            action = 4;
            hp = (TProfile*)oldh1;
         } else {
            if (action < 0) action = -4;
            if (opt.Contains("profs"))
               hp = new TProfile(hname,htitle,gNbins[1],gVmin[1], gVmax[1],"s");
            else
               hp = new TProfile(hname,htitle,gNbins[1],gVmin[1], gVmax[1],"");
            if (!hkeep) {
               hp->SetBit(kCanDelete);
               hp->SetDirectory(0);
            }
         }

         EntryLoop(exprFileName,action,hp,nentries, firstentry, option);

         if (!opt.Contains("goff")) hp->Draw(option);
      } else {
         TH2F *h2;
         if (oldh1) {
            h2 = (TH2F*)oldh1;
         } else {
            h2 = new TH2F(hname,htitle,gNbins[1],gVmin[1], gVmax[1], gNbins[0], gVmin[0], gVmax[0]);
            if (!hkeep) {
               const Int_t kNoStats = BIT(9);
               h2->SetBit(kCanDelete);
               h2->SetBit(kNoStats);
               h2->SetDirectory(0);
            }
         }
         Int_t noscat = strlen(option);
         if (opt.Contains("same")) noscat -= 4;
         if (noscat) {
            EntryLoop(exprFileName,action,h2,nentries, firstentry, option);
 //           if (!fDraw && !opt.Contains("goff")) h2->Draw(option);
            if (!opt.Contains("goff")) h2->Draw(option);
         } else {
            action = 12;
            if (!oldh1 && !opt.Contains("same")) action = -12;
            EntryLoop(exprFileName,action,h2,nentries, firstentry, option);
//            if (oldh1 && !fDraw && !opt.Contains("goff")) h2->Draw(option);
            if (oldh1 && !opt.Contains("goff")) h2->Draw(option);
         }
      }

//*-*- 3-D distribution
   } else if (dimension == 3) {
      action = 13;
      if (!opt.Contains("same")) action = -13;
      EntryLoop(exprFileName,action,0,nentries, firstentry, option);

//*-* an Event List
   //} else if (elist) {
   //   action = 5;
//      Int_t oldEstimate = fEstimate;
//      SetEstimate(1);
   //   EntryLoop(exprFileName,action,elist,nentries, firstentry, option);
//      SetEstimate(oldEstimate);
   }
   delete [] exprFileName;
   delete [] varexp0;
   return gCurrentTableHist;
}

//______________________________________________________________________________
static void FindGoodLimits(Int_t nbins, Int_t &newbins, Float_t &xmin, Float_t &xmax)
{
//*-*-*-*-*-*-*-*-*Find reasonable bin values*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ==========================
//*-*  This mathod is a straight copy of void TTree::FindGoodLimits method
//*-*

   Double_t binlow=0,binhigh=0,binwidth=0;
   Int_t n;
   Double_t dx = 0.1*(xmax-xmin);
   Double_t umin = xmin - dx;
   Double_t umax = xmax + dx;
   if (umin < 0 && xmin >= 0) umin = 0;
   if (umax > 0 && xmax <= 0) umax = 0;

#if ROOT_VERSION_CODE >= ROOT_VERSION(3,03,5)
   THLimitsFinder::Optimize(umin,umax,nbins,binlow,binhigh,n,binwidth,"");
#else
   TGaxis::Optimize(umin,umax,nbins,binlow,binhigh,n,binwidth,"");
#endif

   if (binwidth <= 0 || binwidth > 1.e+39) {
      xmin = -1;
      xmax = 1;
   } else {
      xmin    = binlow;
      xmax    = binhigh;
   }

   newbins = nbins;
}

//______________________________________________________________________________
Bool_t TTable::EntryLoop(const Char_t *exprFileName,Int_t &action, TObject *obj
                          ,Int_t nentries, Int_t firstentry, Option_t *option)
{
 //
 // EntryLoop creates a CINT bytecode to evaluate the given expressions for
 // all table rows in loop and fill the appropriated histograms.
 //
 // Solution for Byte code
 // From: Masaharu Goto <MXJ02154@nifty.ne.jp>
 // To: <fine@bnl.gov>
 // Cc: <rootdev@hpsalo.cern.ch>
 // Sent: 13-th august 1999 year  23:01
 //
 //  action =  1  Fill 1-D histogram obj
 //         =  2  Fill 2-D histogram obj
 //         =  3  Fill 3-D histogram obj
 //         =  4  Fill Profile histogram obj
 //         =  5  Fill a TEventlist
 //         = 11  Estimate Limits
 //         = 12  Fill 2-D PolyMarker obj
 //         = 13  Fill 3-D PolyMarker obj
 //  action < 0   Evaluate Limits for case abs(action)
 //
 //  Load file
   Double_t rmin[3],rmax[3];
   switch(G__loadfile((Char_t *)exprFileName)) {
      case G__LOADFILE_SUCCESS:
      case G__LOADFILE_DUPLICATE:
         break;
      default:
         Error("EntryLoop","Error: loading file %s",exprFileName);
         G__unloadfile((Char_t *)exprFileName);
         return kFALSE; // can not load file
   }

   // Float_t  Selection(Float_t *results[], void *address[], int& i$, int n$)
   //   where  i$ - meta variable to set current row index
   //          n$ - meta variable to set the total table size
   const Char_t *funcName = "SelectionQWERTY";
#define BYTECODE
#ifdef BYTECODE
   const Char_t *argtypes = "Float_t *,float **, int&, int& ";
   Long_t offset;
   G__ClassInfo globals;
   G__MethodInfo func = globals.GetMethod(funcName,argtypes,&offset);

   // Compile bytecode
   struct G__bytecodefunc *pbc = func.GetBytecode();
   if(!pbc) {
      Error("EntryLoop","Bytecode compilation %s",funcName);
      G__unloadfile((Char_t *)exprFileName);
      return kFALSE; // can not get bytecode
   }
#endif
   // Prepare callfunc object
   int i;
   int nRows =  GetNRows();
   TTableDescriptor    *tabsDsc   = GetRowDescriptors();
   tableDescriptor_st  *descTable = tabsDsc->GetTable();
   Float_t  results[]    = {1,1,1,1,1};
   Char_t **addressArray = (Char_t **)new ULong_t[tabsDsc->GetNRows()];
   Char_t *thisTable     = (Char_t *)GetArray();
#ifdef BYTECODE
   G__CallFunc callfunc;
   callfunc.SetBytecode(pbc);

   callfunc.SetArg((long)(&results[0]));   // give 'Float_t *results[5]'     as 1st argument
   callfunc.SetArg((long)(addressArray));  // give 'void    *addressArray[]' as 2nd argument
   callfunc.SetArg((long)(&i));            // give 'int& i$'                 as 3nd argument
   callfunc.SetArg((long)(&nRows));        // give 'int& n$= nRows           as 4th argument
#else
   char buf[200];
   sprintf(buf,"%s((Float_t*)(%ld),(void**)(%ld),*(int*)(%ld),*(int*)(%ld))"
             ,funcName
             ,(long int)results,(long int)addressArray,(long int)(&i),(long int)(&nRows));
#endif

   // Call bytecode in loop

#ifdef BYTECODE
#  define CALLMETHOD callfunc.Exec(0);
#else
#  define CALLMETHOD G__calc(buf);
#endif

#define TAKEACTION_BEGIN                                                                    \
            descTable = tabsDsc->GetTable();                                                \
            for (i=0; i < tabsDsc->GetNRows(); i++,descTable++ )                            \
               addressArray[i] = addressEntry + descTable->fOffset;                         \
            for(i=firstentry;i<lastEntry;i++) {                                             \
            CALLMETHOD

#define TAKEACTION_END  for (int j=0; j < tabsDsc->GetNRows(); j++ ) addressArray[j] += rSize;}


   if (firstentry < nRows ) {
      Long_t rSize         = GetRowSize();
      Char_t *addressEntry = thisTable + rSize*firstentry;
      Int_t lastEntry = TMath::Min(UInt_t(firstentry+nentries),UInt_t(nRows));
      if (action < 0) {
         gVmin[0] = gVmin[1] = gVmin[2] = 1e30;
         gVmax[0] = gVmax[1] = gVmax[2] = -gVmin[0];
      }
      Int_t nchans = 0;
      switch ( action ) {
         case -1: {
            TAKEACTION_BEGIN
            if (results[1]) {
               if (gVmin[0] > results[0]) gVmin[0] = results[0];
               if (gVmax[0] < results[0]) gVmax[0] = results[0];
            }
            TAKEACTION_END

            nchans = gNbins[0];
            if (gVmin[0] >= gVmax[0]) { gVmin[0] -= 1; gVmax[0] += 1;}
            FindGoodLimits(nchans,gNbins[0],gVmin[0],gVmax[0]);
            ((TH1 *)obj)->SetBins(gNbins[0],gVmin[0],gVmax[0]);
         }
         // Intentional fall though
         case  1:
            TAKEACTION_BEGIN
            if (results[1]) ((TH1 *)obj)->Fill(Axis_t(results[0]),Stat_t(results[1]));
            TAKEACTION_END
            gCurrentTableHist = ((TH1 *)obj);
            break;
         case  -2:
            TAKEACTION_BEGIN
            if (results[2]) {
               if (gVmin[0] > results[1]) gVmin[0] = results[1];
               if (gVmax[0] < results[1]) gVmax[0] = results[1];
               if (gVmin[1] > results[0]) gVmin[1] = results[0];
               if (gVmax[1] < results[0]) gVmax[1] = results[0];
            }
            TAKEACTION_END
            nchans = gNbins[0];
            if (gVmin[0] >= gVmax[0]) { gVmin[0] -= 1; gVmax[0] += 1;}
            FindGoodLimits(nchans,gNbins[0],gVmin[0],gVmax[0]);
            if (gVmin[1] >= gVmax[1]) { gVmin[1] -= 1; gVmax[1] += 1;}
            FindGoodLimits(nchans,gNbins[1],gVmin[1],gVmax[1]);
            ((TH1*)obj)->SetBins(gNbins[1],gVmin[1],gVmax[1],gNbins[0],gVmin[0],gVmax[0]);
            // Intentional fall though
         case   2:
            if (obj->IsA() == TH2F::Class()) {
               TAKEACTION_BEGIN
               if (results[2]) ((TH2F*)obj)->Fill(Axis_t(results[0]),Axis_t(results[1]),Stat_t(results[2]));
               TAKEACTION_END
            } else if (obj->IsA() == TH2S::Class()) {
               TAKEACTION_BEGIN
               if (results[2]) ((TH2S*)obj)->Fill(Axis_t(results[0]),Axis_t(results[1]),Stat_t(results[2]));
               TAKEACTION_END
            } else if (obj->IsA() == TH2C::Class()) {
               TAKEACTION_BEGIN
               if (results[2]) ((TH2C*)obj)->Fill(Axis_t(results[0]),Axis_t(results[1]),Stat_t(results[2]));
               TAKEACTION_END
            } else if (obj->IsA() == TH2D::Class()) {
               TAKEACTION_BEGIN
               if (results[2]) ((TH2D*)obj)->Fill(Axis_t(results[0]),Axis_t(results[1]),Stat_t(results[2]));
               TAKEACTION_END
            }
            gCurrentTableHist =  ((TH1 *)obj);
            break;
         case -4:
            TAKEACTION_BEGIN
            if (results[2]) {
               if (gVmin[0] > results[1]) gVmin[0] = results[1];
               if (gVmax[0] < results[1]) gVmax[0] = results[1];
               if (gVmin[1] > results[0]) gVmin[1] = results[0];
               if (gVmax[1] < results[0]) gVmax[1] = results[0];
            }
            TAKEACTION_END
            nchans = gNbins[1];
            if (gVmin[1] >= gVmax[1]) { gVmin[1] -= 1; gVmax[1] += 1;}
            FindGoodLimits(nchans,gNbins[1],gVmin[1],gVmax[1]);
            ((TProfile*)obj)->SetBins(gNbins[1],gVmin[1],gVmax[1]);
            // Intentional fall though
         case  4:
            TAKEACTION_BEGIN
            if (results[2]) ((TProfile*)obj)->Fill(Axis_t(results[0]),Axis_t(results[1]),Stat_t(results[2]));
            TAKEACTION_END
            break;
         case -12:
            TAKEACTION_BEGIN
            if (results[2]) {
               if (gVmin[0] > results[1]) gVmin[0] = results[1];
               if (gVmax[0] < results[1]) gVmax[0] = results[1];
               if (gVmin[1] > results[0]) gVmin[1] = results[0];
               if (gVmax[1] < results[0]) gVmax[1] = results[0];
            }
            TAKEACTION_END
            nchans = gNbins[0];
            if (gVmin[0] >= gVmax[0]) { gVmin[0] -= 1; gVmax[0] += 1;}
            FindGoodLimits(nchans,gNbins[0],gVmin[0],gVmax[0]);
            if (gVmin[1] >= gVmax[1]) { gVmin[1] -= 1; gVmax[1] += 1;}
            FindGoodLimits(nchans,gNbins[1],gVmin[1],gVmax[1]);
            ((TH2F*)obj)->SetBins(gNbins[1],gVmin[1],gVmax[1],gNbins[0],gVmin[0],gVmax[0]);
            // Intentional fall though
         case  12: {
            if (!strstr(option,"same") && !strstr(option,"goff")) {
               ((TH2F*)obj)->DrawCopy(option);
               gPad->Update();
            }
//            pm->SetMarkerStyle(GetMarkerStyle());
//            pm->SetMarkerColor(GetMarkerColor());
//            pm->SetMarkerSize(GetMarkerSize());
            Float_t *x = new Float_t[lastEntry-firstentry]; // pm->GetX();
            Float_t *y = new Float_t[lastEntry-firstentry]; // pm->GetY();
            Float_t u, v;
            Float_t umin = gPad->GetUxmin();
            Float_t umax = gPad->GetUxmax();
            Float_t vmin = gPad->GetUymin();
            Float_t vmax = gPad->GetUymax();
            Int_t pointIndex = 0;
            TAKEACTION_BEGIN
            if (results[2]) {
               u = gPad->XtoPad(results[0]);
               v = gPad->YtoPad(results[1]);
               if (u < umin) u = umin;
               if (u > umax) u = umax;
               if (v < vmin) v = vmin;
               if (v > vmax) v = vmax;
               x[pointIndex] = u;
               y[pointIndex] = v;
               pointIndex++;
            }
            TAKEACTION_END
            if (pointIndex && !strstr(option,"goff")) {
               TPolyMarker *pm = new TPolyMarker(pointIndex,x,y);
               pm->Draw();
               pm->SetBit(kCanDelete);
            }
            if (!((TH2F*)obj)->TestBit(kCanDelete))
               if (pointIndex)
                  for(i=0;i<pointIndex;i++) ((TH2F*)obj)->Fill(x[i], y[i]);
            delete [] x; delete [] y;
            gCurrentTableHist = ((TH1*)obj);
            }
            break;
         case -13:
            TAKEACTION_BEGIN
            if (results[3]) {
               if (gVmin[0] > results[2]) gVmin[0] = results[2];
               if (gVmax[0] < results[2]) gVmax[0] = results[2];
               if (gVmin[1] > results[1]) gVmin[1] = results[1];
               if (gVmax[1] < results[1]) gVmax[1] = results[1];
               if (gVmin[2] > results[0]) gVmin[2] = results[0];
               if (gVmax[2] < results[0]) gVmax[2] = results[0];
            }
            TAKEACTION_END
            rmin[0] = gVmin[2]; rmin[1] = gVmin[1]; rmin[2] = gVmin[0];
            rmax[0] = gVmax[2]; rmax[1] = gVmax[1]; rmax[2] = gVmax[0];
            gPad->Clear();
            gPad->Range(-1,-1,1,1);
            TView::CreateView(1,rmin,rmax);
            // Intentional fall though
         case 13: {
            TPolyMarker3D *pm3d = new TPolyMarker3D(lastEntry-firstentry);
            pm3d->SetBit(kCanDelete);
//            pm3d->SetMarkerStyle(GetMarkerStyle());
//            pm3d->SetMarkerColor(GetMarkerColor());
//            pm3d->SetMarkerSize(GetMarkerSize());
            TAKEACTION_BEGIN
            if (results[3]) pm3d->SetNextPoint(results[0],results[1],results[2]);
            TAKEACTION_END
            pm3d->Draw();
         }
         break;
         default:
            Error("EntryLoop","unknown action \"%d\" for table <%s>", action, GetName());
            break;
      };
   }
   G__unloadfile((Char_t *)exprFileName);
   delete [] addressArray;
   return kTRUE;
}

//______________________________________________________________________________
TTable::TTable(const char *name, Int_t size) : TDataSet(name),
         fSize(size),fN(0), fTable(0),fMaxIndex(0)
{
   // Default TTable ctor.
   if (size == 0) Warning("TTable(0)","Wrong table format");
}

//______________________________________________________________________________
TTable::TTable(const char *name, Int_t n,Int_t size) : TDataSet(name),
        fSize(size),fN(0),fTable(0),fMaxIndex(0)
{
   // Create TTable object and set array size to n longs.
   if (n > 0) Set(n);
}

//______________________________________________________________________________
TTable::TTable(const char *name, Int_t n, Char_t *table,Int_t size) : TDataSet(name),
         fSize(size),fN(0),fTable(0),fMaxIndex(0)
{
   // Create TTable object and initialize it with values of array.

   Set(n, table);
}

//______________________________________________________________________________
TTable::TTable(const char *name, const char *type, Int_t n, Char_t *array, Int_t size)
         : TDataSet(name),fSize(size),fTable(0),fMaxIndex(0)
{
   // Create TTable object and initialize it with values of array.

   fTable = array;
   SetType(type);
   SetfN(n);
}

//______________________________________________________________________________
TTable::TTable(const TTable &table):TDataSet(table)
{
   // Copy constructor.
   fTable    = 0;
   SetUsedRows(table.GetNRows());
   fSize     = table.GetRowSize();
   SetfN(table.fN);
   Set(table.fN, table.fTable);
}

//______________________________________________________________________________
TTable &TTable::operator=(const TTable &rhs)
{
   // TTable assignment operator.
   // This operator REALLOCATEs this table to fit the number of
   // the USED rows of the source table if any

   if (strcmp(GetType(),rhs.GetType()) == 0) {
      if (this != &rhs && rhs.GetNRows() >0 ){
         Set(rhs.GetNRows(), rhs.fTable);
         SetUsedRows(rhs.GetNRows());
      }
   } else
      Error("operator=","Can not copy <%s> table into <%s> table", rhs.GetType(),GetType());
   return *this;
}

//______________________________________________________________________________
TTable::~TTable()
{
   // Delete TTable object.
   Delete();
}

//______________________________________________________________________________
void TTable::Adopt(Int_t n, void *arr)
{
   // Adopt array arr into TTable, i.e. don't copy arr but use it directly
   // in TTable. User may not delete arr, TTable dtor will do it.

   Clear();

   SetfN(n); SetUsedRows(n);
   fTable = (char *)arr;
}

//______________________________________________________________________________
Int_t TTable::AddAt(const void *row)
{
  // Add        the "row" at the GetNRows() position, and
  // reallocate the table if neccesary,               and
  // return     the row index the "row" has occupied.
  //
  // row == 0 see method TTable::AddAt(const void *row, Int_t i)

   Int_t gap = GetTableSize() - GetNRows();
   // do we need to add an extra space?
   if (gap < 1) ReAllocate(GetTableSize() + TMath::Max(1,Int_t(0.3*GetTableSize())));
   Int_t indx = GetNRows();
   AddAt(row,indx);
   return indx;
}
//______________________________________________________________________________
void TTable::AddAt(const void *row, Int_t i)
{
   // Add    one element ("row") of structure at position "i".
   // Check  for out of bounds.
   //
   //        If the row == 0 the "i" cell is still occupied and
   // filled with the pattern "ff"

   if (!BoundsOk("TTable::AddAt", i))
      i = 0;
   if (row) memcpy(fTable+i*fSize,row,fSize);
   else memset(fTable+i*fSize,127,fSize);
   SetUsedRows(TMath::Max((Int_t)i+1,Int_t(fMaxIndex)));
}

//______________________________________________________________________________
void TTable::CopyStruct(Char_t *dest, const Char_t *src)
{
 // Copy the C-structure src into the new location
 // the length of the strucutre is defined by this class descriptor
   ::memcpy(dest,src,fSize*fN);
}
//______________________________________________________________________________
void TTable::CopySet(TTable &array)
{
   //to be documented
   array.Set(fN);
   CopyStruct(array.fTable,fTable);
}
//______________________________________________________________________________
const Char_t *TTable::GetColumnComment(Int_t columnIndex) const {
   // Get a comment from the table descriptor
   TDataSetIter nextComment(GetRowDescriptors()->MakeCommentField(kFALSE));
   TDataSet *nxc = 0;
   for (int i=0; i<= columnIndex; i++) nxc = nextComment();
   return nxc ? nxc->GetTitle() : 0;
}
//______________________________________________________________________________
Long_t TTable::AppendRows(const void *row, UInt_t nRows)
{
   // Append nRows row of the array "row" to the table
   // return
   //    - the new table size (# of table rows)
   //    - 0 if the object doesn't own the internal array and can not expand it
   if (!TestBit(kIsNotOwn) && row && nRows ) {
      Int_t indx = GetNRows();
      ReAllocate(nRows);
      // Copy (insert) the extra staff in
      ::memmove(fTable+indx*fSize,row,fSize*nRows);
   }
   return TestBit(kIsNotOwn) ? 0 : GetSize();
}
//______________________________________________________________________________
Long_t TTable::InsertRows(const void *row, Long_t indx, UInt_t nRows)
{
  // void InsertRows(cons void *row, Long_t indx, UInt_t nRows)
  //
  // Insert one or several rows into the table at "indx" position
  // The rest table stuff is shifted down
  //
  //  cons void    - a pointer to the array of rows to be inserted
  //  Long_t indx =  The position these rows will be inserted to
  //  Int_t nRows  - the total number of rows to be inserted
  //                 = 1 "by default
  //  return:
  //  The number of the rows has been shifted to accomodate
  //  the new rows.
  //
   Long_t nShifted = 0;
   if (nRows > 0) {
      // Shift the table down
      nShifted = CopyRows(this, indx, indx+nRows, GetNRows()+nRows);
      // Copy (insert) the extra staff in
      ::memmove(fTable+indx*fSize,row,fSize*nRows);
   }
   return nShifted;

}
//______________________________________________________________________________
void *TTable::ReAllocate()
{
  // Reallocate this table leaving only (used rows)+1 allocated
  // GetTableSize() = GetNRows() + 1
  // returns a pointer to the first row of the reallocated table
  // Note:
  // The table is reallocated if it is an owner of the internal array

   ReAlloc(GetNRows()+1);
   return (void *)fTable;
}
//______________________________________________________________________________
void *TTable::ReAllocate(Int_t newsize)
{
  // Reallocate this table leaving only <newsize> allocated
  // GetTableSize() = newsize;
  // returns a pointer to the first row of the reallocated table
  // Note:
  // The table is reallocated if it is an owner of the internal array

   if (newsize > fN) ReAlloc(newsize);
   return (void *)fTable;
}

//______________________________________________________________________________
void TTable::ReAlloc(Int_t newsize)
{
  // The table is reallocated if it is an owner of the internal array
   if (!TestBit(kIsNotOwn) && newsize > 0) {
      void *arr = 0;
      Int_t sleepCounter = 0;
      while (!(arr =  realloc(fTable,fSize*newsize))) {
         sleepCounter++;
         Warning("ReAlloc",
              "Not enough memory to Reallocate %d bytes for table <%s::%s>. Please cancel some jobs",
              newsize, GetType(),GetName());
         gSystem->Sleep(1000*600);
         if (sleepCounter > 30) {
            Error("ReAlloc","I can not wait anymore. Good bye");
            assert(0);
         }
      }
      SetfN(newsize);
      fTable = (char *)arr;
   }
}

//______________________________________________________________________________
Char_t *TTable::Create()
{
   // Allocate a space for the new table, if any
   // Sleep for a while if space is not available and try again
   if (!fTable) {
      void *ptr = 0;
      Int_t sleepCounter = 0;
      while (!(ptr = malloc(fSize*fN))) {
         sleepCounter++;
         Warning("Create",
            "Not enough memory to allocate %d rows for table <%s::%s>. Please cancel some jobs",
            fN, GetType(),GetName());
         gSystem->Sleep(1000*600);
         if (sleepCounter > 30){
            Error("Create","I can not wait anymore. Good bye");
            assert(0);
         }
      }
      fTable = (Char_t *)ptr;
      // make sure all link-columns are zero
      memset(fTable,0,fSize*fN);
   }
   return fTable;
}

//______________________________________________________________________________
void TTable::Browse(TBrowser *b){
   // Wrap each table coulumn with TColumnView object to browse.
   if (!b) return;
   TDataSet::Browse(b);
   Int_t nrows = TMath::Min(Int_t(GetNRows()),6);
   if (nrows == 0) nrows = 1;
   Print(0,nrows);
   // Add the table columns to the browser
   UInt_t nCol = GetNumberOfColumns();
   for (UInt_t i = 0;i<nCol;i++){
      TColumnView *view = 0;
      UInt_t nDim = GetDimensions(i);
      const Char_t *colName = GetColumnName(i);
      if (!nDim) { // scalar
         // This will cause a small memory leak
         // unless TBrowser recognizes kCanDelete bit
         if( GetColumnType(i)== kPtr) {
            UInt_t offset = GetOffset(i);
            TTableMap *m = *(TTableMap **)(((char *)GetArray())+offset);
            if (m) {
               TString nameMap = "*";
               nameMap += m->Table()->GetName();
               b->Add(m,nameMap.Data());
            }
         } else {
            view = new TColumnView(GetColumnName(i),this);
            view->SetBit(kCanDelete);
            b->Add(view,view->GetName());
         }
      } else {     // array
         const UInt_t *indx = GetIndexArray(i);
         UInt_t totalSize = 1;
         UInt_t k;
         for (k=0;k<nDim; k++) totalSize *= indx[k];
         for (k=0;k<totalSize;k++) {
            TString buffer;
            buffer.Form("%s[%d]",colName,k);
            view = new TColumnView(buffer,this);
            view->SetBit(kCanDelete);
            b->Add(view,view->GetName());
         }
      }
   }
}

//______________________________________________________________________________
void TTable::Clear(Option_t *opt)
{
   // Deletes the internal array of this class
   // if this object does own its internal table

   if (!fTable) return;
   Bool_t dtor = kFALSE;
   dtor = opt && (strcmp(opt,gDtorName)==0);
   if (!opt || !opt[0] || dtor ) {
      if (! TestBit(kIsNotOwn)) {
         if (!dtor) ResetMap();
         free(fTable);
      }
      fTable    = 0;
      fMaxIndex = 0;
      SetfN(0);
      return;
   }
}

//______________________________________________________________________________
void TTable::Delete(Option_t *opt)
{
   //
   // Delete the internal array and free the memory it occupied
   // if this object did own this array
   //
   // Then perform TDataSet::Delete(opt)
   Clear(gDtorName);
   TDataSet::Delete(opt);
}

//______________________________________________________________________________
TClass  *TTable::GetRowClass() const
{
   //to be documented
   TClass *cl = 0;
   TTableDescriptor *dsc = GetRowDescriptors();
   if (dsc) cl = dsc->RowClass();
   else Error("GetRowClass()","Table descriptor of <%s::%s> table lost",
             GetName(),GetType());
   return cl;
}

//______________________________________________________________________________
Long_t TTable::GetNRows() const {
// Returns the number of the used rows for the wrapped table
   return fMaxIndex;
}

//______________________________________________________________________________
Long_t TTable::GetRowSize() const {
// Returns the size (in bytes) of one table row
   return fSize;
}

//______________________________________________________________________________
Long_t TTable::GetTableSize() const {
// Returns the number of the allocated rows
   return fN;
}

//______________________________________________________________________________
void TTable::Fit(const char *formula ,const char *varexp, const char *selection,Option_t *option ,Option_t *goption,Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Fit a projected item(s) from a TTable*-*-*-*-*-*-*-*-*-*
//*-*              =======================================
//
//  formula is a TF1 expression.
//
//  See TTable::Draw for explanations of the other parameters.
//
//  By default the temporary histogram created is called htemp.
//  If varexp contains >>hnew , the new histogram created is called hnew
//  and it is kept in the current directory.
//  Example:
//    table.Fit(pol4,"sqrt(x)>>hsqrt","y>0")
//    will fit sqrt(x) and save the histogram as "hsqrt" in the current
//    directory.
//

   TString opt(option);
   opt += "goff";

   Draw(varexp,selection,opt,nentries,firstentry);

   TH1 *hfit = gCurrentTableHist;
   if (hfit) {
      Printf("hname=%s, formula=%s, option=%s, goption=%s\n",hfit->GetName(),formula,option,goption);
      // remove bit temporary
      Bool_t canDeleteBit = hfit->TestBit(kCanDelete);
      if (canDeleteBit)  hfit->ResetBit(kCanDelete);
      hfit->Fit(formula,option,goption);
      if (TestBit(canDeleteBit))   hfit->SetBit(kCanDelete);
   }
   else      Printf("ERROR hfit=0\n");
}

//______________________________________________________________________________
const Char_t *TTable::GetType() const
{
//Returns the type of the wrapped C-structure kept as the TNamed title
   return GetTitle();
}

//______________________________________________________________________________
Bool_t TTable::IsFolder() const {
   // return Folder flag to be used by TBrowse object
   // The table is a folder if
   //  - it has sub-dataset
   //    or
   //  - GetNRows > 0
   return kTRUE; // to provide the "fake" folder bit to workaround TKey::Browse()

#if 0
   // this became useless due TKey::Browse new implementation
   return
   (fList && fList->Last() ? kTRUE : kFALSE)
   ||
     (GetNRows() > 0);
#endif
}

//______________________________________________________________________________
Int_t TTable::NaN()
{
//
// return the total number of the NaN for float/double cells of this table
// Thanks Victor Perevoztchikov
//

   EColumnType code;
   char const *cell,*colname,*table;
   double word;
   int icol,irow,colsize,wordsize,nwords,iword,nerr,offset;

   TTableDescriptor *rowDes = GetRowDescriptors();
   assert(rowDes!=0);
   table = (const char*)GetArray();

   int ncols = rowDes->GetNumberOfColumns();

   int lrow  = GetRowSize();
   int nrows = GetNRows  ();
   nerr =0;
   for (icol=0; icol < ncols; icol++) {// loop over cols
      code = rowDes->GetColumnType(icol);
      if (code!=kFloat && code!=kDouble) continue;

      offset   = rowDes->GetOffset    (icol);
      colsize  = rowDes->GetColumnSize(icol);
      wordsize = rowDes->GetTypeSize  (icol);
      nwords = colsize/wordsize;
      for (irow=0; irow < nrows; irow++) { //loop over rows
         cell = table + offset + irow*lrow;
         for (iword=0;iword<nwords; iword++,cell+=wordsize) { //words in col
            word = (code==kDouble) ? *(double*)cell : *(float*)cell;
            if (TMath::Finite(word))     continue;
//              ERROR FOUND
            nerr++; colname = rowDes->GetColumnName(icol);
            Warning("NaN"," Table %s.%s.%d\n",GetName(),colname,irow);
         }
      }
   }
   return nerr;
}

//______________________________________________________________________________
TTable *TTable::New(const Char_t *name, const Char_t *type, void *array, UInt_t size)
{
  // This static method creates a new TTable object if provided

   TTable *table = 0;
   if (type && name) {
      TString tableType(type);
      TString t = tableType.Strip();

      TString classname("St_");
      classname += t;
      TClass *cl = TClass::GetClass(classname);
      if (cl) {
         table = (TTable *)cl->New();
         if (table) {
            table->SetTablePointer(array);
            table->SetName(name);
            table->SetfN(size);
            table->SetUsedRows(size);
         }
      }
   }
   return table;
}
//______________________________________________________________________________
Bool_t TTable::OutOfBoundsError(const char *where, Int_t i) const
{
   // Generate an out-of-bounds error. Always returns false.
   Error(where, "index %d out of bounds (size: %d, this: 0x%lx)", i, fN, (Long_t)this);
   return kFALSE;
}
//______________________________________________________________________________
Char_t *TTable::Print(Char_t *strbuf,Int_t lenbuf) const
{
   // Create IDL table defintion (to be used for XDF I/O)
   Int_t iOut = 0;

   TTableDescriptor *dscT = GetRowDescriptors();
   if (!dscT ) {
      Error("Print"," No dictionary entry for <%s> structure", GetTitle());
      if (lenbuf>0) iOut += snprintf(strbuf,lenbuf," *** Errror ***");
      return strbuf;
   }

   TROOT::IndentLevel();
   if (lenbuf>0) {
      // cut of the "_st" suffix
      Char_t *typenam =  new Char_t [strlen(dscT->GetName())+1];
      strlcpy(typenam,dscT->GetName(),strlen(dscT->GetName())+1);
      // look for the last "_"
      Char_t *last = strrchr(typenam,'_');
      // Check whether it is "_st"
      Char_t *eon = 0;
      if (last) eon = strstr(last,"_st");
      // Cut it off if any
      if (eon) *eon = '\0';
      iOut += snprintf(strbuf+iOut,lenbuf-iOut,"struct %s {",typenam);
      delete [] typenam;
   } else {
      cout << "struct " << dscT->GetName() << " {" << endl;
   }

   TTableDescriptor::iterator dsc  = dscT->begin();
   TTableDescriptor::iterator dscE = dscT->end();
   TDataSetIter nextComment(dscT->MakeCommentField(kFALSE));
   for (;dsc != dscE; dsc++) {
      TROOT::IndentLevel();
      TString name = GetTypeName(EColumnType((*dsc).fType));
      if (lenbuf>0) {
         // convert C type names to CORBA type names
         name.ReplaceAll("unsigned char","octet");
         name.ReplaceAll("int","long");
         iOut += snprintf(strbuf+iOut,lenbuf-iOut," %s %s",name.Data(),(*dsc).fColumnName);
      } else
         cout << '\t'<< name.Data() << '\t'<< (*dsc).fColumnName;

      Int_t indx;
      Int_t dim = (*dsc).fDimensions;
      for  (indx = 0; indx < dim; indx++) {
         if (lenbuf>0)
            iOut += snprintf(strbuf+iOut,lenbuf-iOut,"[%d]",(*dsc).fIndexArray[indx]);
         else
            cout <<  "[" << dec << (*dsc).fIndexArray[indx]<<"]";
      }
      // print comment if any
      TDataSet *nxc = nextComment();
      if (lenbuf>0)
         iOut += snprintf(strbuf+iOut,lenbuf-iOut, ";");
      else {
         const char *title = nxc ? nxc->GetTitle() : " ";
         cout << ";\t//" << title << endl;
      }
   } /* dsc */

   TROOT::IndentLevel();
   if (lenbuf>0)
      iOut += snprintf(strbuf+iOut,lenbuf-iOut, "}");
   else
      cout << "}" << endl;
   return strbuf;
}

//______________________________________________________________________________
const Char_t *TTable::PrintHeader() const
{
  // Print general table inforamtion
   cout << endl << " ---------------------------------------------------------------------------------------" << endl
        <<  " " << Path()
                <<"  Allocated rows: "<<fN
                <<"\t Used rows: "<<fMaxIndex
                <<"\t Row size: "      << fSize << " bytes"
        <<endl;
   return 0;
}

//______________________________________________________________________________
const Char_t *TTable::Print(Int_t row, Int_t rownumber, const Char_t *, const Char_t *) const
{
  //const Char_t *TTable::Print(Int_t row, Int_t rownumber, const Char_t *colfirst, const Char_t *collast) const
  //
  //  Print the contents of internal table per COLUMN.
  //
  //  row       - the index of the first row to print (counting from ZERO)
  //  rownumber - the total number of rows to print out (=10 by default)
  //
  //  (No use !) Char_t *colfirst, *collast - the names of the first/last
  //                                          to print out (not implemented yet)
  //
  //--------------------------------------------------------------
   // Check bounds and adjust it
   Int_t const width = 8;
   Int_t rowStep = 10; // The maximun values to print per line
   Int_t rowNumber = rownumber;
   if (row  > Int_t(GetSize()) || GetSize() == UInt_t(0))  {
      PrintHeader();
      cout  << " ======================================================================================" << endl
           << "   There are " << GetSize() << " allocated rows for this table only"                     << endl
             << " ======================================================================================" << endl;
      return 0;
   }
   if (rowNumber > Int_t(GetSize()-row)) rowNumber = GetSize()-row;
   if (!rowNumber) return 0;
   rowStep = TMath::Min(rowStep,rowNumber);

   Int_t cdate = 0;
   Int_t ctime = 0;
   UInt_t *cdatime = 0;
   Bool_t isdate = kFALSE;

   TTableDescriptor *dscT = GetRowDescriptors();
   if (!dscT ) return 0;

   //  3. Loop by "rowStep x lines"

   const Char_t  *startRow = (const Char_t *)GetArray() + row*GetRowSize();
   Int_t rowCount = rowNumber;
   Int_t thisLoopLenth = 0;
   const Char_t  *nextRow = 0;
   while (rowCount) {
      PrintHeader();
      if  (GetNRows() == 0) {// to Print empty table header
         cout  << " ======================================================================================" << endl
               << "   There is NO filled row in this table"                                                 << endl
               << " ======================================================================================" << endl;
         return 0;
      }
      cout << " Table: " << dscT->GetName()<< "\t";
      for (Int_t j = row+rowNumber-rowCount; j<row+rowNumber-rowCount+rowStep && j < row+rowNumber ;j++) {
         Int_t hW = width-2;
         if (j>=10) hW -= (int)TMath::Log10(float(j))-1;
         cout  << setw(hW) << "["<<j<<"]";
         cout  << " :" ;
      }
      cout << endl
      <<       " ======================================================================================" << endl;
      TTableDescriptor::iterator member = dscT->begin();
      TTableDescriptor::iterator   dscE = dscT->end();
      TDataSetIter nextComment(dscT->MakeCommentField(kFALSE));

      for (; member != dscE; member++){
         TString membertype = GetTypeName(EColumnType((*member).fType));
         isdate = kFALSE;
         if (strcmp((*member).fColumnName,"fDatime") == 0 && membertype == "UInt_t")
                                                                                   isdate = kTRUE;
         cout << membertype.Data();

         // Add the dimensions to "array" members
         Int_t dim = (*member).fDimensions;
         Int_t indx = 0;
         UInt_t *arrayLayout = 0;
         if (dim) {
            arrayLayout = new UInt_t[dim];
            memset(arrayLayout,0,dim*sizeof(Int_t));
         }
         Int_t arrayLength  = 1;
         while (indx < dim ){ // Take in account the room this index will occupy
            arrayLength *= (*member).fIndexArray[indx];
            indx++;
         }
         // Encode data value or pointer value
         Int_t offset = (*member).fOffset;
         Int_t thisStepRows;
         thisLoopLenth = TMath::Min(rowCount,rowStep);
         Int_t indexOffset;
         Bool_t breakLoop = kFALSE;

         for (indexOffset=0; indexOffset < arrayLength && !breakLoop; indexOffset++) {
            nextRow = startRow;

            if (!indexOffset) cout << "\t" << (*member).fColumnName;
            else              cout << "\t" << setw(strlen((*member).fColumnName)) << " ";

            if (dim) {
               for (Int_t i=0;i<dim;i++) cout << "["<<dec<<arrayLayout[i]<<"]";
               ArrayLayout(arrayLayout,(*member).fIndexArray,dim);
            }
            cout << "\t";
            if ( strlen((*member).fColumnName)+3*dim < 8) cout << "\t";

            for (thisStepRows = 0;thisStepRows < thisLoopLenth; thisStepRows++,nextRow += GetRowSize()) {
               const char *pointer = nextRow + offset  + indexOffset*(*member).fTypeSize;
               if (isdate) {
                  cdatime = (UInt_t*)pointer;
                  TDatime::GetDateTime(cdatime[0],cdate,ctime);
                  cout << cdate << "/" << ctime;
               } else if ((*member).fType == kChar && dim == 1) {
                  char charbuffer[11];
                  strlcpy(charbuffer,pointer,TMath::Min(10,arrayLength)+1);
                  charbuffer[10] = 0;
                  cout << "\"" << charbuffer;
                  if (arrayLength > 10)
                     cout << " . . . ";
                  cout << "\"";
                  breakLoop = kTRUE;
               } else {
                  AsString((void *)pointer,EColumnType((*member).fType),width,cout);
                  cout << " :";
               }
            }
            // Encode  the column's comment
            if (indexOffset==0) {
               TDataSet *nxc = nextComment();
               cout << " " << (const char *)(nxc ? nxc->GetTitle() : "no comment");
            }
            cout << endl;
         }
         if (arrayLayout) delete [] arrayLayout;
      }
      rowCount -= thisLoopLenth;
      startRow  = nextRow;
   }
   cout << "---------------------------------------------------------------------------------------" << endl;
   return 0;
}
//______________________________________________________________________________
void TTable::PrintContents(Option_t *) const
{
   //to be documented
   TDataSet::PrintContents();
   TROOT::IndentLevel();
   Printf("\tclass %s: public TTable\t --> Allocated rows: %d\t Used rows: %d\t Row size: %d bytes\n",
         IsA()->GetName(),int(fN),int(fMaxIndex),int(fSize));

}

//______________________________________________________________________________
void TTable::Project(const char *hname, const char *varexp, const char *selection, Option_t *option,Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Make a projection of a TTable using selections*-*-*-*-*-*-*
//*-*              =============================================
//
//   Depending on the value of varexp (described in Draw) a 1-D,2-D,etc
//   projection of the TTable will be filled in histogram hname.
//   Note that the dimension of hname must match with the dimension of varexp.
//

   TString var;
   var.Form("%s>>%s",varexp,hname);

   TString opt(option);
   opt += "goff";

   Draw(var,selection,opt,nentries,firstentry);
}

//______________________________________________________________________________
Int_t TTable::Purge(Option_t *opt)
{
   // Shrink the table to free the unused but still allocated rows
   ReAllocate();
   return TDataSet::Purge(opt);
}

//______________________________________________________________________________
void TTable::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
//   Save a primitive as a C++ statement(s) on output stream "out".
   UInt_t arrayLayout[10],arraySize[10];
   const unsigned char *pointer=0,*startRow=0;
   int i,rowCount;unsigned char ic;

   out << "TDataSet *CreateTable() { " << endl;

   Int_t rowNumber =  GetNRows();
   TTableDescriptor *dscT = GetRowDescriptors();

//                      Is anything Wrong??
   if (!rowNumber || !dscT ) {//
      out << "// The output table was bad-defined!" << endl
          << " fprintf(stderr, \"Bad table found. Please remove me\\n\");" << endl
          << " return 0; } "    << endl;
      return;
   }

   startRow = (const UChar_t *)GetArray();
   assert(startRow!=0);

   const Char_t *rowId = "row";
   const Char_t *tableId = "tableSet";

//                      Generate the header

   const char *className = IsA()->GetName();

   out << "// -----------------------------------------------------------------" << endl;
   out << "// "   << Path()
       << " Allocated rows: "<< rowNumber
       <<"  Used rows: "<<      rowNumber
       <<"  Row size: " << fSize << " bytes"                 << endl;
   out << "// "  << " Table: " << dscT->GetName()<<"[0]--> "
       << dscT->GetName()<<"["<<rowNumber-1 <<"]"            << endl;
   out << "// ====================================================================" << endl;
   out << "// ------  Test whether this table share library was loaded ------"      << endl;
   out << "  if (!TClass::GetClass(\"" << className << "\")) return 0;"    << endl;
   out <<    dscT->GetName() << " " << rowId << ";" << endl
       <<  className << " *" << tableId << " = new "
       <<  className
       << "(\""<<GetName()<<"\"," << GetNRows() << ");" << endl
       << "//" <<endl ;

//              Row loop
   TDataSetIter nextComment(dscT->MakeCommentField(kFALSE));
   for (rowCount=0;rowCount<rowNumber; rowCount++,startRow += fSize, nextComment.Reset()) {     //row loop
      out << "memset(" << "&" << rowId << ",0," << tableId << "->GetRowSize()" << ");" << endl ;

//              Member loop
   TTableDescriptor::iterator member  = dscT->begin();
   TTableDescriptor::iterator   dscE  = dscT->end();
   for (; member != dscE; member++) {  //LOOP over members
      TString memberType = GetTypeName(EColumnType((*member).fType));
      TString memberName((*member).fColumnName);

       // Encode  the column's comment
      TDataSet *nxc = nextComment();
      TString memberTitle(nxc ? nxc->GetTitle() : "no comment");

      Int_t offset = (*member).fOffset;
      int mayBeName = 0;
      if (memberName.Index("name",0,TString::kIgnoreCase)>=0) mayBeName=1999;
      if (memberName.Index("file",0,TString::kIgnoreCase)>=0) mayBeName=1999;
      int typeSize = (*member).fTypeSize;

//              Add the dimensions to "array" members
      Int_t dim = (*member).fDimensions;
      if (dim) memset(arrayLayout,0,dim*sizeof(Int_t));
      Int_t arrayLength  = 1;
      for (int indx=0;indx < dim ;indx++){
         arraySize[indx] =  (*member).fIndexArray[indx];;
         arrayLength *= arraySize[indx];
      }

//                      Special case, character array
      int charLen = (memberType.CompareTo("char")==0);
      if (charLen) {    //Char case
         charLen=arrayLength;
         pointer = startRow + offset;
//                      Actual size of char array
         if (mayBeName) {
            charLen = strlen((const char*)pointer)+1;
            if (charLen>arrayLength) charLen = arrayLength;
         } else {
            for(;charLen && !pointer[charLen-1];charLen--){;}
            if (!charLen) charLen=1;
         }

         out << " memcpy(&" << rowId << "." << (const char*)memberName;
         out << ",\"";
         for (int ii=0; ii<charLen;ii++) {
            ic = pointer[ii];
            if (ic && (isalnum(ic)
             || strchr("!#$%&()*+-,./:;<>=?@{}[]_|~",ic))) {//printable
               out << ic;
            } else {                                      //nonprintable
               out << "\\x" << setw(2) << setfill('0') << hex << (unsigned)ic ;
               out << setw(1) << setfill(' ') << dec;
            }
         }
         out << "\"," << dec << charLen << ");";
         out << "// " << (const char*)memberTitle << endl;
         continue;
      } //EndIf of char case

//                      Normal member
      Int_t indexOffset;
      for (indexOffset=0; indexOffset < arrayLength ; indexOffset++) {//array loop
         out << setw(3) << " " ;
         out << " " << rowId << "." << (const char*)memberName;

         if (dim) {
            for (i=0;i<dim;i++) {out << "["<<dec<<arrayLayout[i]<<"]";}
            ArrayLayout(arrayLayout,arraySize,dim);}

//                      Generate "="
            out << "\t = ";

            pointer = startRow + offset  + indexOffset*typeSize;

            AsString((void *)pointer,EColumnType((*member).fType),10,out);

//                      Encode data member title
            if (indexOffset==0)  out << "; // " << (const char*)memberTitle;
            out << ";" << endl;
         }//end array loop
      }//end of member loop

      out << tableId << "->AddAt(&" << rowId <<");" << endl;

   }//end of row loop
   out << "// ----------------- end of code ---------------" << endl
       << " return (TDataSet *)tableSet;" << endl
       << "}"  << endl;
   return;
}

//______________________________________________________________________________
void TTable::Set(Int_t n)
{
   // Set array size of TTable object to n longs. If n<0 leave array unchanged.
   if (n < 0) return;
   if (fN != n)  Clear();
   SetfN(n);
   if (fN == 0) return;
   Create();
   if (TTable::GetNRows()) Reset();
}
//______________________________________________________________________________
void TTable::SetTablePointer(void *table)
{
   //to be documented
   if (fTable) free(fTable);
   fTable = (Char_t *)table;
}

//______________________________________________________________________________
void TTable::SetType(const char *const type)
{
   //to be documented
   SetTitle(type);
}

//______________________________________________________________________________
static Char_t *GetExpressionFileName()
{
   // Create a name of the file in the temporary directory if any
   const Char_t *tempDirs =  gSystem->Getenv("TEMP");
   if (!tempDirs)  tempDirs =  gSystem->Getenv("TMP");
   if (!tempDirs) tempDirs = "/tmp";
   if (gSystem->AccessPathName(tempDirs)) tempDirs = ".";
   if (gSystem->AccessPathName(tempDirs)) return 0;
   TString fileName;
   fileName.Form("Selection.C.%d.tmp",gSystem->GetPid());
   return  gSystem->ConcatFileName(tempDirs,fileName.Data());
}

//______________________________________________________________________________
Char_t *TTable::MakeExpression(const Char_t *expressions[],Int_t nExpressions)
{
  // Create CINT macro to evaluate the user-provided expresssion
  // Expression may contains:
  //   -  the table columen names
  //   - 2 meta names: i$ - the current column index,
  //                   n$ - the total table size provided by TTable::GetNRows() method
  //
  // return the name of temporary file with the current expressions
  //
   const Char_t *typeNames[] = {"NAN","float", "int",  "long",  "short",         "double"
                                ,"unsigned int","unsigned long", "unsigned short","unsigned char"
                                ,"char", "TTableMap &"};
   const char *resID     = "results";
   const char *addressID = "address";
   Char_t *fileName = GetExpressionFileName();
   if (!fileName) {
      Error("MakeExpression","Can not create a temporary file");
      return 0;
   }

   ofstream str;
   str.open(fileName);
   if (str.bad() ) {
      Error("MakeExpression","Can not open the temporary file <%s>",fileName);
      delete [] fileName;
      return 0;
   }

   TTableDescriptor *dsc = GetRowDescriptors();
   const tableDescriptor_st *descTable  = dsc->GetTable();
   // Create function
   str << "void SelectionQWERTY(float *"<<resID<<", float **"<<addressID<< ", int& i$, int& n$ )"   << endl;
   str << "{"                                                        << endl;
   int i = 0;
   for (i=0; i < dsc->GetNRows(); i++,descTable++ ) {
      // Take the column name
      const Char_t *columnName = descTable->fColumnName;
      const Char_t *type = 0;
      // First check whether we do need this column
      for (Int_t exCount = 0; exCount < nExpressions; exCount++) {
         if (expressions[exCount] && expressions[exCount][0] && strstr(expressions[exCount],columnName)) goto LETSTRY;
      }
      continue;
LETSTRY:
      Bool_t isScalar = !(descTable->fDimensions);
      Bool_t isFloat = descTable->fType == kFloat;
      type = typeNames[descTable->fType];
      str << type << " ";
      if (!isScalar)  str << "*";

      str << columnName << " = " ;
      if (isScalar)   str << "*(";
      if (!isFloat)   str << "(" << type << "*)";
      str << addressID << "[" << i << "]";
      if (isScalar)   str << ")" ;
      str << ";" << endl;
   }
   // Create expressions
   for (i=0; i < nExpressions; i++ ) {
      if (expressions[i] && expressions[i][0])
         str << " "<<resID<<"["<<i<<"]=(float)(" << expressions[i] << ");"  << endl;
//      if (i == nExpressions-1 && i !=0 )
//          str  << "  if ("<<resID<<"["<<i<<"] == 0){ return; }" << endl;
   };
   str << "}" << endl;
   str.close();
   // Create byte code and check syntax
   if (str.good()) return fileName;
   delete [] fileName;
   return 0;
}

//______________________________________________________________________________
void TTable::Reset(Int_t c)
{
   // Fill the entire table with byte "c" ;
   ///     c=0 "be default"
   if (fTable) {
      ResetMap(kTRUE);
      ::memset(fTable,c,fSize*fN);
      if (c) ResetMap(kFALSE);
   }
}

//______________________________________________________________________________
void TTable::ResetMap(Bool_t wipe)
{
   // Clean all filled columns with the pointers to TTableMap
   // if any
   //  wipe = kTRUE - delete all object the Map's point to
   //         kFALSE - zero pointer, do not call "delete" though
   piterator links     = pbegin();
   piterator lastLinks = pend();
   for (;links != lastLinks;links++) {
      TTableMap **mp = (TTableMap **)(*links);
      if (wipe) delete *mp;
      *mp = 0;
   }
}
//______________________________________________________________________________
void TTable::Set(Int_t n, Char_t *array)
{
   // Set array size of TTable object to n longs and copy array.
   // If n<0 leave array unchanged.

   if (n < 0) return;
   if (fN < n) Clear();

   SetfN(n);

   if (fN == 0) return;
   Create();
   CopyStruct(fTable,array);
   fMaxIndex = n;
}

//_______________________________________________________________________
void TTable::StreamerTable(TBuffer &b,Version_t version)
{
   // Stream an object of class TTable.
   if (b.IsReading()) {
      TDataSet::Streamer(b);
      b >> fN;
      StreamerHeader(b,version);
      //   Create a table to fit nok rows
      Set(fMaxIndex);
   } else {
      TDataSet::Streamer(b);
      b << fN;
      StreamerHeader(b,version);
   }
}

//_______________________________________________________________________
void TTable::StreamerHeader(TBuffer &b, Version_t version)
{
   // Read "table parameters first"
   if (b.IsReading()) {
      Long_t rbytes;
      if (version) { }   // version to remove compiler warning
#ifdef __STAR__
      if (version < 3) {
         // skip obsolete  STAR fields (for the sake of the backward compatibility)
         //   char name[20];   /* table name */
         //   char type[20];   /* table type */
         //   long maxlen;     /* # rows allocated */
         long len = b.Length() + (20+4) + (20+4) + 4;
         b.SetBufferOffset(len);
      }
#endif
      b >> fMaxIndex;         // fTableHeader->nok;          /* # rows filled */
      b >> rbytes;            /* number of bytes per row */
      if (GetRowSize() == -1) fSize = rbytes;
      if (rbytes - GetRowSize()) {
         Warning("StreamerHeader","Schema evolution warning: row size mismatch: expected %ld, read %ld bytes\n",GetRowSize(),rbytes);
      }

#ifdef __STAR__
      if (version < 3) {
         // skip obsolete  STAR fields (for the sake of the backward compatibility)
         //    long dsl_pointer;  /* swizzled (DS_DATASET_T*) */
         //    long data_pointer; /* swizzled (char*) */
         long len = b.Length() + (4) + (4);
         b.SetBufferOffset(len);
      }
#endif
   } else {
      b << fMaxIndex;         //fTableHeader->nok;          /* # rows filled */
      b << fSize;             //  fTableHeader->rbytes;     /* number of bytes per row */
   }
}
//_______________________________________________________________________
Int_t TTable::SetfN(Long_t len)
{
   //to be documented
   fN = len;
   return fN;
}
//____________________________________________________________________________
#ifdef StreamElelement
#define __StreamElelement__ StreamElelement
#undef StreamElelement
#endif

#define StreamElementIn(type)  case TTableDescriptor::_NAME2_(k,type):        \
 if (evolutionOn) {                                  \
     if (nextCol->fDimensions)  {                    \
       if (nextCol->fOffset != UInt_t(-1)) {         \
          R__b.ReadFastArray((_NAME2_(type,_t) *)(row+nextCol->fOffset),nextCol->fSize/sizeof(_NAME2_(type,_t)));   \
       } else {                                        \
           _NAME2_(type,_t) *readPtrV = new _NAME2_(type,_t)[nextCol->fSize/sizeof(_NAME2_(type,_t))];              \
           R__b.ReadFastArray((_NAME2_(type,_t) *)(row+nextCol->fOffset),nextCol->fSize/sizeof(_NAME2_(type,_t)));  \
           delete [] readPtrV;                       \
           readPtrV = 0;                             \
       }                                             \
     }                                               \
     else  {                                         \
       _NAME2_(type,_t) skipBuffer;                  \
       _NAME2_(type,_t) *readPtr =  (_NAME2_(type,_t) *)(row+nextCol->fOffset); \
       if (nextCol->fOffset == UInt_t(-1)) readPtr = &skipBuffer;               \
       R__b >> *readPtr;                             \
     }                                               \
 } else {                                            \
   if (nextCol->fDimensions)  {                      \
     R__b.ReadFastArray  ((_NAME2_(type,_t) *)(row+nextCol->fOffset),nextCol->fSize/sizeof(_NAME2_(type,_t)));  \
   } else                                                       \
     R__b >> *(_NAME2_(type,_t) *)(row+nextCol->fOffset);       \
 }                                                              \
 break

#define StreamElementOut(type) case TTableDescriptor::_NAME2_(k,type):    \
 if (nextCol->fDimensions)                                    \
    R__b.WriteFastArray((_NAME2_(type,_t) *)(row+nextCol->fOffset), nextCol->fSize/sizeof(_NAME2_(type,_t))); \
 else                                                         \
    R__b << *(_NAME2_(type,_t) *)(row+nextCol->fOffset);      \
 break

//______________________________________________________________________________
TTableDescriptor  *TTable::GetRowDescriptors() const
{
   //to be documented
   TTableDescriptor *dsc = 0;
   if (IsA()) dsc = GetDescriptorPointer();
   if (!dsc) {
      Error("GetRowDescriptors()","%s has no dictionary !",GetName());
      dsc = GetTableDescriptors();
      ((TTableDescriptor *)this)->SetDescriptorPointer(dsc);
   }
   return dsc;
}
//______________________________________________________________________________
TTableDescriptor *TTable::GetDescriptorPointer() const
{
   //to be documented
   assert(0);
   return 0;
}

//______________________________________________________________________________
void TTable::SetDescriptorPointer(TTableDescriptor *)
{
   //to be documented
   assert(0);
}

//______________________________________________________________________________
void TTable::Streamer(TBuffer &R__b)
{
   // Stream an array of the "plain" C-structures
   TTableDescriptor *ioDescriptor = GetRowDescriptors();
   TTableDescriptor *currentDescriptor = ioDescriptor;
   Version_t R__v = 0;
   if (R__b.IsReading()) {
      // Check whether the file is the "obsolete" one
      R__v = R__b.ReadVersion();
      Bool_t evolutionOn = kFALSE;
      if (R__v>=2) {
         if (IsA() != TTableDescriptor::Class()) {
            if (R__v>3) {
               R__b >> ioDescriptor;
            } else {  // backward compatibility
               ioDescriptor =  new TTableDescriptor();
               ioDescriptor->Streamer(R__b);
            }
            if (!currentDescriptor) {
               currentDescriptor = ioDescriptor;
               SetDescriptorPointer(currentDescriptor);
            }
            if (currentDescriptor->fSecondDescriptor != ioDescriptor) {
               // Protection against of memory leak.
               delete currentDescriptor->fSecondDescriptor;
               currentDescriptor->fSecondDescriptor = ioDescriptor;
            }

            // compare two descriptors
            evolutionOn = (Bool_t)ioDescriptor->UpdateOffsets(currentDescriptor);
         }
      }
      TTable::StreamerTable(R__b,R__v);
      if (fMaxIndex <= 0) return;
      char *row= fTable;
      Int_t maxColumns = ioDescriptor->NumberOfColumns();
      Int_t rowSize = GetRowSize();
      if (evolutionOn) Reset(0); // Clean table
      for (Int_t indx=0;indx<fMaxIndex;indx++,row += rowSize) {
         tableDescriptor_st *nextCol = ioDescriptor->GetTable();
         for (Int_t colCounter=0; colCounter < maxColumns; colCounter++,nextCol++) {
            // Stream one table row supplied
            switch(nextCol->fType) {
               StreamElementIn(Float);
               StreamElementIn(Int);
               StreamElementIn(Long);
               StreamElementIn(Short);
               StreamElementIn(Double);
               StreamElementIn(UInt);
               StreamElementIn(ULong);
               StreamElementIn(UChar);
               StreamElementIn(Char);
               StreamElementIn(Bool);
               case TTableDescriptor::kPtr: {
                  Ptr_t readPtr;
                  R__b >> readPtr;
                  if (evolutionOn) {
                     // TTableMap skipBuffer;
                     //  R__b >> readPtr;
                     if (nextCol->fOffset == UInt_t(-1)) delete readPtr; // skip this member
                     else *(Ptr_t *)(row+nextCol->fOffset) = readPtr;
                  } else {
                     *(Ptr_t *)(row+nextCol->fOffset) = readPtr;
               }
               break;
                                       }
               default:
               break;
            };
         }
      }
   } else {
      TSeqCollection *save = fList;
      R__b.WriteVersion(TTable::IsA());

      //      if (Class_Version()==2)
      if (IsA() != TTableDescriptor::Class()) {
         if ( Class_Version()>3 ) {
            R__b << ioDescriptor;
         } else {  // backward compatibility
            ioDescriptor->Streamer(R__b);
         }
      } else {
         if ( Class_Version()<=3 ) fList = 0;
      }

      TTable::StreamerTable(R__b);
      if (fMaxIndex <= 0) return;
      char *row= fTable;
      Int_t maxColumns = ioDescriptor->NumberOfColumns();
      Int_t rowSize = GetRowSize();
      for (Int_t indx=0;indx<fMaxIndex;indx++,row += rowSize) {
         tableDescriptor_st *nextCol = ioDescriptor->GetTable();
         for (Int_t colCounter=0; colCounter < maxColumns; colCounter++,nextCol++) {
            // Stream one table row supplied
            switch(nextCol->fType) {
               StreamElementOut(Float);
               StreamElementOut(Int);
               StreamElementOut(Long);
               StreamElementOut(Short);
               StreamElementOut(Double);
               StreamElementOut(UInt);
               StreamElementOut(ULong);
               StreamElementOut(UChar);
               StreamElementOut(Char);
               StreamElementOut(Bool);
               case TTableDescriptor::kPtr:
                  R__b << *(Ptr_t *)(row+nextCol->fOffset);
                  break;
               default:
                  break;
            };
         }
      }
      fList = save;
   }
}
#ifdef __StreamElelement__
#define StreamElelement __StreamElelement__
#undef __StreamElelement__
#endif

//_______________________________________________________________________
void TTable::Update()
{
   //to be documented
}

//_______________________________________________________________________
void TTable::Update(TDataSet *set, UInt_t opt)
{
 // Kill the table current data
 // and adopt those from set
   if (set->HasData()) {
      // Check whether the new table has the same type
      if (strcmp(GetTitle(),set->GetTitle()) == 0 ) {
         TTable *table =  (TTable *)set;
         Adopt(table->GetSize(),table->GetArray());
         // Adopt can not distniguish the "allocated" and "used"
         // rows,
         // correct the corrupted number of the "used" rows
         SetUsedRows(table->GetNRows());
         // mark that object lost the STAF table and can not delete it anymore
         table->SetBit(kIsNotOwn);
         // mark we took over of this STAF table
         ResetBit(kIsNotOwn);
      } else
         Error("Update",
             "This table is <%s> but the updating one has a wrong type <%s>",GetTitle(),set->GetTitle());
   }
   TDataSet::Update(set,opt);
}
//_______________________________________________________________________
const char *TTable::TableDictionary(const char *className,const char *structName,TTableDescriptor *&ColDescriptors)
{
   // Query the TClass instance for the C-stucture dicitonary
   // This method is to be used  with TableImp CPP macro (see $ROOTSYS/table/inc/Ttypes.h
   if (className){/*NotUsed*/};
   TClass *r = TClass::GetClass(structName,1);
   ColDescriptors = new TTableDescriptor(r);
   return structName;
}


 //  ----   Table descriptor service   ------
Int_t        TTable::GetColumnIndex(const Char_t *columnName) const {return GetRowDescriptors()->ColumnByName(columnName);}
const Char_t *TTable::GetColumnName(Int_t columnIndex) const {return GetRowDescriptors()->ColumnName(columnIndex); }
const UInt_t *TTable::GetIndexArray(Int_t columnIndex) const {return GetRowDescriptors()->IndexArray(columnIndex); }
UInt_t       TTable::GetNumberOfColumns()              const {return GetRowDescriptors()->NumberOfColumns();       }

UInt_t       TTable::GetOffset(Int_t columnIndex)      const {return GetRowDescriptors()->Offset(columnIndex); }
Int_t        TTable::GetOffset(const Char_t *columnName) const {return GetRowDescriptors()->Offset(columnName); }

UInt_t       TTable::GetColumnSize(Int_t columnIndex)  const {return GetRowDescriptors()->ColumnSize(columnIndex); }
Int_t        TTable::GetColumnSize(const Char_t *columnName) const {return GetRowDescriptors()->ColumnSize(columnName); }

UInt_t       TTable::GetTypeSize(Int_t columnIndex)    const {return GetRowDescriptors()->TypeSize(columnIndex); }
Int_t        TTable::GetTypeSize(const Char_t *columnName) const {return GetRowDescriptors()->TypeSize(columnName); }

UInt_t       TTable::GetDimensions(Int_t columnIndex)  const {return GetRowDescriptors()->Dimensions(columnIndex); }
Int_t        TTable::GetDimensions(const Char_t *columnName) const {return GetRowDescriptors()->Dimensions(columnName); }

TTable::EColumnType  TTable::GetColumnType(Int_t columnIndex)  const {return GetRowDescriptors()->ColumnType(columnIndex); }
TTable::EColumnType  TTable::GetColumnType(const Char_t *columnName) const {return GetRowDescriptors()->ColumnType(columnName); }

//  pointer iterator
//________________________________________________________________________________________________________________
TTable::piterator::piterator(const TTable *t,EColumnType type): fCurrentRowIndex(0),fCurrentColIndex(0),fRowSize(0),fCurrentRowPtr(0),fCurrentColPtr(0)
{
   //to be documented
   Int_t sz = 0;
   if (t) sz = t->GetNRows();
   if (sz) {
      fRowSize       = t->GetRowSize();
      fCurrentRowPtr = (const Char_t *)t->GetArray();

      TTableDescriptor    *tabsDsc       = t->GetRowDescriptors();
      TTableDescriptor::iterator ptr     = tabsDsc->begin();
      TTableDescriptor::iterator lastPtr = tabsDsc->end();
      UInt_t i =0;
      for( i = 0; ptr != lastPtr; ptr++,i++)
         if ( tabsDsc->ColumnType(i) == type ) fPtrs.push_back(tabsDsc->Offset(i));
      if (fPtrs.size()==0) {
         MakeEnd(t->GetNRows());
      } else {
         column();
      }
   } else {
      MakeEnd(0);
   }
} // piterator(TTable *)

