// @(#)root/graf:$Name:  $:$Id: TLatex.cxx,v 1.15 2001/05/31 15:45:11 rdm Exp $
// Author: Nicolas Brun   07/08/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdio.h>

#include "TROOT.h"
#include "TLatex.h"
#include "TVirtualPad.h"
#include "TVirtualPS.h"
#include "TArc.h"

#ifdef R__SUNCCBUG
const Double_t kPI = 3.14159265358979323846;
#else
const Double_t kPI = TMath::Pi();
#endif
const Int_t kLatex = BIT(10);

ClassImp(TLatex)

//______________________________________________________________________________
//
//   TLatex : to draw Mathematical Formula
//
//   This class has been implemented by begin_html <a href="http://pcbrun.cern.ch/nicolas/index.html">Nicolas Brun</a> end_html.
//   ========================================================
//
//   TLatex's purpose is to write mathematical equations
//   The syntax is very similar to the Latex one :
//
//   ** Subscripts and Superscripts
//   ------------------------------
//   Subscripts and superscripts are made with the _ and ^ commands.  These commands
//   can be combined to make complicated subscript and superscript expressions.
//   You may choose how to display subscripts and superscripts using the 2 functions
//   SetIndiceSize(Float_t) and SetLimitIndiceSize(Int_t).
//Begin_Html
/*p
<img src="gif/latex_subscripts.gif">
*/
//End_Html
//
//   ** Fractions
//   ------------
//   Fractions denoted by the / symbol are made in the obvious way.
//   The #frac command is used for large fractions in displayed formula; it has
//   two arguments: the numerator and the denominator.
//Begin_Html
/*
<img src="gif/latex_frac.gif">
*/
//End_Html
//
//   ** Roots
//   --------
//   The #sqrt command produces the square root of its argument; it has an optional
//   first argument for other roots.
//   ex: #sqrt{10}  #sqrt[3]{10}
//
//   ** Mathematical Symbols
//   -----------------------
//   TLatex can make dozens of special mathematical symbols. A few of them, such as
//   + and > , are produced by typing the corresponding keyboard character.  Others
//   are obtained with the commands in the following table :
//Begin_Html
/*
<img src="gif/latex_symbols.gif">
*/
//End_Html
//    #Box draw a square
//
//   ** Delimiters
//   -------------
//   You can produce 3 kinds of proportional delimiters.
//   #[]{....} or "a la" Latex #left[.....#right] : big square brackets
//   #{}{....} or              #left{.....#right} : big curly brackets
//   #||{....} or              #left|.....#right| : big absolute value symbol
//   #(){....} or              #left(.....#right) : big parenthesis
//
//   ** Greek Letters
//   ----------------
//   The command to produce a lowercase Greek letter is obtained by adding a # to
//   the name of the letter. For an uppercase Greek letter, just capitalize the first
//   letter of the command name.
//   #alpha #beta #gamma #delta #epsilon #zeta #eta #theta #iota #kappa #lambda #mu
//   #nu #xi #omicron #pi #varpi #rho #sigma #tau #upsilon #phi #varphi #chi #psi #omega
//   #Gamma #Delta #Theta #Lambda #Xi #Pi #Sigma #Upsilon #Phi #Psi #Omega
//
//   ** Putting One Thing Above Another
//   ----------------------------------
//   Symbols in a formula are sometimes placed on above another. TLatex provides
//   special commands for doing this.
//
//   ** Accents
//   ----------
//    #hat{a} = hat
//    #check  = inversed hat
//    #acute  = acute
//    #grave  = agrave
//    #dot    = derivative
//    #ddot   = double derivative
//
//Begin_Html
/*
<img src="gif/latex_above.gif">
*/
//End_Html
//   #dot  #ddot  #hat  #check  #acute  #grave
//
//   ** Changing Style in Math Mode
//   ------------------------------
//   You can change the font and the text color at any moment using :
//   #font[font-number]{...} and #color[color-number]{...}
//
//   ** Example1
//   -----------
//     The following macro (tutorials/latex.C) produces the following picture:
//  {
//     gROOT->Reset();
//     TCanvas c1("c1","Latex",600,700);
//     TLatex l;
//     l.SetTextAlign(12);
//     l.SetTextSize(0.04);
//     l.DrawLatex(0.1,0.8,"1)   C(x) = d #sqrt{#frac{2}{#lambdaD}}  #int^{x}_{0}cos(#frac{#pi}{2}t^{2})dt");
//     l.DrawLatex(0.1,0.6,"2)   C(x) = d #sqrt{#frac{2}{#lambdaD}}  #int^{x}cos(#frac{#pi}{2}t^{2})dt");
//     l.DrawLatex(0.1,0.4,"3)   R = |A|^{2} = #frac{1}{2}(#[]{#frac{1}{2}+C(V)}^{2}+#[]{#frac{1}{2}+S(V)}^{2})");
//     l.DrawLatex(0.1,0.2,"4)   F(t) = #sum_{i=-#infty}^{#infty}A(i)cos#[]{#frac{i}{t+i}}");
//  }
//Begin_Html
/*
<img src="gif/latex_example.gif">
*/
//End_Html
//
//   ** Example2
//   -----------
//     The following macro (tutorials/latex2.C) produces the following picture:
//  {
//     gROOT->Reset();
//     TCanvas c1("c1","Latex",600,700);
//     TLatex l;
//     l.SetTextAlign(23);
//     l.SetTextSize(0.1);
//     l.DrawLatex(0.5,0.95,"e^{+}e^{-}#rightarrowZ^{0}#rightarrowI#bar{I}, q#bar{q}");
//     l.DrawLatex(0.5,0.75,"|#vec{a}#bullet#vec{b}|=#Sigmaa^{i}_{jk}+b^{bj}_{i}");
//     l.DrawLatex(0.5,0.5,"i(#partial_{#mu}#bar{#psi}#gamma^{#mu}+m#bar{#psi}=0#Leftrightarrow(#Box+m^{2})#psi=0");
//     l.DrawLatex(0.5,0.3,"L_{em}=eJ^{#mu}_{em}A_{#mu} , J^{#mu}_{em}=#bar{I}#gamma_{#mu}I , M^{j}_{i}=#SigmaA_{#alpha}#tau^{#alphaj}_{i}");
//  }
//Begin_Html
/*
<img src="gif/latex_example2.gif">
*/
//End_Html
//
//   ** Example3
//   -----------
//     The following macro (tutorials/latex3.C) produces the following picture:
//  {
//     gROOT->Reset();
//   TCanvas c1("c1");
//   TPaveText pt(.1,.5,.9,.9);
//   pt.AddText("#frac{2s}{#pi#alpha^{2}}  #frac{d#sigma}{dcos#theta} (e^{+}e^{-} #rightarrow f#bar{f} ) = ");
//   pt.AddText("#left| #frac{1}{1 - #Delta#alpha} #right|^{2} (1+cos^{2}#theta");
//   pt.AddText("+ 4 Re #left{ #frac{2}{1 - #Delta#alpha} #chi(s) #[]{#hat{g}_{#nu}^{e}#hat{g}_{#nu}^{f}
//    (1 + cos^{2}#theta) + 2 #hat{g}_{a}^{e}#hat{g}_{a}^{f} cos#theta) } #right}");
//   pt.SetLabel("Born equation");
//   pt.Draw();
//  }
//Begin_Html
/*
<img src="gif/latex_example3.gif">
*/
//End_Html
//
//   ** Alignment rules
//   ------------------
//  The TText alignment rules apply to the TLatex objects with one exception
//  concerning the vertical alignment:
//  If the vertical alignment = 1 , subscripts are not taken into account
//  if the vertical alignment = 0 , the text is aligned to the box surrounding
//                                  the full text with sub and superscripts
//  This is illustrated in the following example:
//
//{
//  gROOT->Reset();
//  TCanvas c1("c1","c1",600,500);
//  c1.SetGrid();
//  c1.DrawFrame(0,0,1,1);
//  const char *longstring = "K_{S}... K^{*0}... #frac{2s}{#pi#alpha^{2}}
// #frac{d#sigma}{dcos#theta} (e^{+}e^{-} #rightarrow f#bar{f} ) =
// #left| #frac{1}{1 - #Delta#alpha} #right|^{2} (1+cos^{2}#theta)";
//
//  TLatex latex;
//  latex.SetTextSize(0.033);
//  latex.SetTextAlign(13);  //align at top
//  latex.DrawLatex(.2,.9,"K_{S}");
//  latex.DrawLatex(.3,.9,"K^{*0}");
//  latex.DrawLatex(.2,.8,longstring);
//
//  latex.SetTextAlign(12);  //centered
//  latex.DrawLatex(.2,.6,"K_{S}");
//  latex.DrawLatex(.3,.6,"K^{*0}");
//  latex.DrawLatex(.2,.5,longstring);
//
//  latex.SetTextAlign(11);  //default bottom alignment
//  latex.DrawLatex(.2,.4,"K_{S}");
//  latex.DrawLatex(.3,.4,"K^{*0}");
//  latex.DrawLatex(.2,.3,longstring);
//
//  latex.SetTextAlign(10);  //special bottom alignment
//  latex.DrawLatex(.2,.2,"K_{S}");
//  latex.DrawLatex(.3,.2,"K^{*0}");
//  latex.DrawLatex(.2,.1,longstring);
//
//  latex.SetTextAlign(12);
//  latex->SetTextFont(72);
//  latex->DrawLatex(.1,.80,"13");
//  latex->DrawLatex(.1,.55,"12");
//  latex->DrawLatex(.1,.35,"11");
//  latex->DrawLatex(.1,.18,"10");
//}
//Begin_Html
/*
<img src="gif/latex_alignment.gif">
*/
//End_Html
//______________________________________________________________________________


//______________________________________________________________________________
TLatex::TLatex()
{
// default constructor
      fFactorSize  = 1.5;
      fFactorPos   = 0.6;
      fLimitFactorSize = 3;
      fError       = 0;
      fShow        = kFALSE;
      fPos=fTabMax = 0;
      SetLineWidth(2);
}

//______________________________________________________________________________
TLatex::TLatex(Double_t x, Double_t y, const char *text)
       :TText(x,y,text)
{
// normal constructor
      fFactorSize  = 1.5;
      fFactorPos   = 0.6;
      fLimitFactorSize = 3;
      fError       = 0;
      fShow        = kFALSE;
      fPos=fTabMax = 0;
      SetLineWidth(2);
}

//______________________________________________________________________________
TLatex::~TLatex()
{
}

//______________________________________________________________________________
TLatex::TLatex(const TLatex &text)
{
   ((TLatex&)text).Copy(*this);
}

//______________________________________________________________________________
void TLatex::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*Copy this TLatex object to another TLatex*-*-*-*-*-*-*
//*-*                  =========================================

   ((TLatex&)obj).fFactorSize  = fFactorSize;
   ((TLatex&)obj).fFactorPos   = fFactorPos;
   ((TLatex&)obj).fLimitFactorSize  = fLimitFactorSize;
   ((TLatex&)obj).fError       = fError;
   ((TLatex&)obj).fShow        = fShow;
   ((TLatex&)obj).fOriginSize  = fOriginSize;
   ((TLatex&)obj).fTabMax      = fTabMax;
   ((TLatex&)obj).fPos         = fPos;
   TText::Copy(obj);
   TAttLine::Copy(((TAttLine&)obj));
}

//______________________________________________________________________________
FormSize TLatex::Anal1(TextSpec_t spec, const Char_t* t, Int_t length)
{
   return Analyse(0,0,spec,t,length);
}


//______________________________________________________________________________
FormSize TLatex::Analyse(Double_t x, Double_t y, TextSpec_t spec, const Char_t* t, Int_t length)
{
//  Analyse and paint the TLatex formula
//
//  It is called twice : first for calculating the size of
//  each portion of the formula, then to paint the formula.
//  When analyse finds an operator or separator, it calls
//  itself recursively to analyse the arguments of the operator.
//  when the argument is an atom (normal text), it calculates
//  the size of it and return it as the result.
//  for example : if the operator #frac{arg1}{arg2} is found :
//  Analyse(arg1) return the size of arg1 (width, up, down)
//  Analyse(arg2) return the size of arg2
//  now, we know the size of #frac{arg1}{arg2}  :
//  width = max(width_arg1, width_arg2)
//  up = up_arg1 + down_arg1
//  down = up_arg2 + down_arg2
//  so, when the user wants to paint a fraction at position (x,y),
//  the rect used for the formula is : (x,y-up,x+width,y+down)
//
// return size of zone occupied by the text/formula
// t : chain to be analyzed
// length : number of chars in t.
//
const char *tab[] = { "alpha","beta","chi","delta","epsilon","phi","gamma","eta","iota","varphi","kappa","lambda",
                "mu","nu","omicron","pi","theta","rho","sigma","tau","upsilon","varpi","omega","xi","psi","zeta",
                "varpi","varpi","varpi","Delta","varpi","Phi","Gamma","varpi","varpi","varpi",
                "varpi","Lambda","varpi","varpi","varpi","Pi","Theta","varpi","Sigma","varpi",
                "Upsilon","varpi","Omega","Xi","Psi" };

const char *tab2[] = { "leq","/","infty","voidb","club","diamond","heart",
                 "spade","leftrightarrow","leftarrow","uparrow","rightarrow",
                 "downarrow","circ","pm","doublequote","geq","times","propto",
                 "partial","bullet","divide","neq","equiv","approx","3dots",
                 "cbar","topbar","downleftarrow","aleph","Jgothic","Rgothic","voidn",
                 "otimes","oplus","oslash","cap","cup","supset","supseteq",
                 "notsubset","subset","subseteq","in","notin","angle","nabla",
                 "oright","ocopyright","trademark","prod","surd","upoint","corner","wedge",
                 "vee","Leftrightarrow","Leftarrow","Uparrow","Rightarrow",
                 "Downarrow","diamond","LT","void1","copyright","void3","sum",
                 "arctop","lbar","arcbottom","topbar","void8", "bottombar","arcbar",
                 "ltbar","void04","void05","void06","GT","int" };

const char *tab3[] = { "bar","vec","dot","hat","ddot","acute","grave","check"};

      if (fError != 0) return FormSize(0,0,0);

      Int_t NbBlancDeb=0,NbBlancFin=0,l_NbBlancDeb=0,l_NbBlancFin=0;
      Int_t i,k;
      Int_t min=0, max=0;
      Bool_t cont = kTRUE;
      while(cont) {
         // count leading blanks
         //while(NbBlancDeb+NbBlancFin<length && t[NbBlancDeb]==' ') NbBlancDeb++;

         if (NbBlancDeb==length) return FormSize(0,0,0); // empty string

         // count trailing blanks
         //while(NbBlancDeb+NbBlancFin<length && t[length-NbBlancFin-1]==' ') NbBlancFin++;

         if (NbBlancDeb==l_NbBlancDeb && NbBlancFin==l_NbBlancFin) cont = kFALSE;

         // remove characters { }
         if (t[NbBlancDeb]=='{' && t[length-NbBlancFin-1]=='}') {
            Int_t NbBrackets = 0;
            Bool_t sameBrackets = kTRUE;
            for(i=NbBlancDeb;i<length-NbBlancFin;i++) {
               if (t[i] == '{' && !(i>0 && t[i-1] == '@')) NbBrackets++;
               if (t[i] == '}' && t[i-1]!= '@') NbBrackets--;
               if (NbBrackets==0 && i<length-NbBlancFin-2) {
                  sameBrackets=kFALSE;
                  break;
               }
            }

            if (sameBrackets) {
               // begin and end brackets match
               NbBlancDeb++;
               NbBlancFin++;
               if (NbBlancDeb+NbBlancFin==length) return FormSize(0,0,0); // empty string
               cont = kTRUE;
            }

         }

         l_NbBlancDeb = NbBlancDeb;
         l_NbBlancFin = NbBlancFin;
      }

      // make a copy of the current processed chain of characters
      // removing leading and trailing blanks
      length -= NbBlancFin+NbBlancDeb; // length of string without blanks
      Char_t* text = new Char_t[length+1];
      strncpy(text,t+NbBlancDeb,length);
      text[length] = 0;

      // compute size of subscripts and superscripts
      Float_t IndiceSize = spec.size/fFactorSize;
      if(IndiceSize<fOriginSize/TMath::Exp(fLimitFactorSize*TMath::Log(fFactorSize))-0.001f)
         IndiceSize = spec.size;
      // substract 0.001 because of rounding errors
      TextSpec_t specNewSize = spec;
      specNewSize.size       = IndiceSize;

      // recherche des operateurs
      Int_t OpPower         = -1;   // Position of first ^ (power)
      Int_t OpUnder         = -1;   // Position of first _ (indice)
      Int_t OpFrac          = -1;   // Position of first \frac
      Int_t OpSqrt          = -1;   // Position of first \sqrt
      Int_t NbBrackets      = 0;    // Nesting level in { }
      Int_t NbCroch         = 0;    // Nesting level in [ ]
      Int_t OpCurlyCurly    = -1;   // Position of first }{
      Int_t OpSquareCurly   = -1;   // Position of first ]{
      Int_t OpCloseCurly    = -2;   // Position of first }
      Int_t OpColor         = -1;   // Position of first \color
      Int_t OpFont          = -1;   // Position of first \font
      Int_t OpGreek         = -1;   // Position of a Greek letter
      Int_t OpSpec          = -1;   // position of a special character
      Int_t OpAbove         = -1;   // position of a vector/overline
      Int_t OpSquareBracket = 0 ;   // position of a "[]{" operator (#[]{arg})
      Int_t OpBigCurly      = 0 ;   // position of a "{}{" operator (big curly bracket #{}{arg})
      Int_t OpAbs           = 0 ;   // position of a "||{" operator (absolute value) (#||{arg})
      Int_t OpParen         = 0 ;   // position of a "(){" operator (big parenthesis #(){arg})
      Int_t AbovePlace      = 0 ;   // true if subscripts must be written above and not after
      Int_t OpBox           = 0 ;   // position of #Box
      Int_t Operp           = 0;    // position of #perp
      Int_t OpOdot          = 0;    // position of #odot
      Int_t Oparallel       = 0;    // position of #parallel
      Bool_t OpFound = kFALSE;
      Bool_t quote1 = kFALSE, quote2 = kFALSE ;

      for(i=0;i<length;i++) {
         switch (text[i]) {
            case '\'' : quote1 = !quote1 ; break ;
            case '"'  : quote2  = !quote2 ; break ;
         }
         //if (quote1 || quote2) continue ;
         switch (text[i]) {
         case '{': if (NbCroch==0) {
                      if (!(i>0 && text[i-1] == '@')) NbBrackets++;
                   }
                   break;
         case '}': if (NbCroch==0) {
                      if (!(i>0 && text[i-1] == '@')) NbBrackets--;
              /*    if (NbBrackets<0) {  marthe
                     // more "}" than "{"
                     fError = "Missing \"{\"";
                     return FormSize(0,0,0);
                  }*/
                     if (NbBrackets==0) {
                       if (i<length-1) if (text[i+1]=='{' && OpCurlyCurly==-1) OpCurlyCurly=i;
                       if (i<length-2) {
                          if (text[i+1]!='{' && !(text[i+2]=='{' && (text[i+1]=='^' || text[i+1]=='_'))
                              && OpCloseCurly==-2) OpCloseCurly=i;
                       }
                       else if (i<length-1) {
                           if (text[i+1]!='{' && OpCloseCurly==-2) OpCloseCurly=i;
                       }
                       else if (OpCloseCurly==-2) OpCloseCurly=i;
                     }
                 }
                 break;
         case '[': if (NbBrackets==0) {
                      if (!(i>0 && text[i-1] == '@')) NbCroch++;
                   }
                   break;
         case ']': if (NbBrackets==0) {
                      if (!(i>0 && text[i-1] == '@')) NbCroch--;
                      if (NbCroch<0) {
                     // more "]" than "["
                        fError = "Missing \"[\"";
                        return FormSize(0,0,0);
                      }
                   }
                   break;
         }
         if (length>i+1) {
            Char_t buf[2];
            strncpy(buf,&text[i],2);
            if (strncmp(buf,"^{",2)==0) {
               if (OpPower==-1 && NbBrackets==0 && NbCroch==0) OpPower=i;
               if (i>3) {
                  Char_t buf[4];
                  strncpy(buf,&text[i-4],4);
                  if (strncmp(buf,"#int",4)==0) AbovePlace = 1;
                  if (strncmp(buf,"#sum",4)==0) AbovePlace = 2;
               }
            }
            if (strncmp(buf,"_{",2)==0) {
               if (OpUnder==-1 && NbBrackets==0 && NbCroch==0) OpUnder=i;
               if (i>3) {
                  Char_t buf[4];
                  strncpy(buf,&text[i-4],4);
                  if (strncmp(buf,"#int",4)==0) AbovePlace = 1;
                  if (strncmp(buf,"#sum",4)==0) AbovePlace = 2;
               }
            }
            if (strncmp(buf,"]{",2)==0)
               if (OpSquareCurly==-1 && NbBrackets==0 && NbCroch==0) OpSquareCurly=i;
         }
         // detect other operators
         if (text[i]=='\\' || text[i]=='#' && !OpFound && NbBrackets==0 && NbCroch==0) {

            if (length>i+8 ) {
               Char_t buf[8];
               strncpy(buf,&text[i+1],8);
               if (!Oparallel && strncmp(buf,"parallel",8)==0) {
                  Oparallel=1; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
            }
            if (length>i+6) {
               Char_t buf[6];
               strncpy(buf,&text[i+1],6);
               if (strncmp(buf,"color[",6)==0 || strncmp(buf,"color{",6)==0) {
                  OpColor=i; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue ;
               }
            }
            if (length>i+5 ) {
               Char_t buf[5];
               strncpy(buf,&text[i+1],5);
               if (strncmp(buf,"frac{",5)==0) {
                  OpFrac=i; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
               if (strncmp(buf,"sqrt{",5)==0 || strncmp(buf,"sqrt[",5)==0) {
                  OpSqrt=i; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
               if (strncmp(buf,"font{",5)==0 || strncmp(buf,"font[",5)==0) {
                  OpFont=i; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
            }
            if (length>i+4 ) {
               Char_t buf[4];
               strncpy(buf,&text[i+1],4);
               if (!OpOdot && strncmp(buf,"odot",4)==0) {
                  OpOdot=1; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
               if (!Operp && strncmp(buf,"perp",4)==0) {
                  Operp=1; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
            }
            if (length>i+3) {
               Char_t buf[3];
               strncpy(buf,&text[i+1],3);
               if (strncmp(buf,"[]{",3)==0) {
                  OpSquareBracket=1; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                   continue;   }
               if (strncmp(buf,"{}{",3)==0 ) {
                  OpBigCurly=1; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
               if (strncmp(buf,"||{",3)==0) {
                  OpAbs=1; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
               if (strncmp(buf,"(){",3)==0) {
                  OpParen=1; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
               if (!OpBox && strncmp(buf,"Box",3)==0) {
                  OpBox=1; OpFound = kTRUE;
                  if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  continue;
               }
            }
            for(k=0;k<51;k++) {
               if (!OpFound && UInt_t(length)>i+strlen(tab[k])) {
                  if (strncmp(&text[i+1],tab[k],strlen(tab[k]))==0) {
                     OpGreek=k;
                     OpFound = kTRUE;
                     if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  }
               }
            }
            for(k=0;k<8;k++) {
               if (!OpFound && UInt_t(length)>i+strlen(tab3[k])) {
                  if (strncmp(&text[i+1],tab3[k],strlen(tab3[k]))==0) {
                     OpAbove=k;
                     OpFound = kTRUE;
                     if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  }
               }
            }
            UInt_t lastsize = 0;
            if (!OpFound)
            for(k=0;k<80;k++) {
               if ((OpSpec==-1 || strlen(tab2[k])>lastsize) && UInt_t(length)>i+strlen(tab2[k])) {
                  if (strncmp(&text[i+1],tab2[k],strlen(tab2[k]))==0) {
                     lastsize = strlen(tab2[k]);
                     OpSpec=k;
                     OpFound = kTRUE;
                     if (i>0 && OpCloseCurly==-2) OpCloseCurly=i-1;
                  }
               }
            }
         }
      }

  /*    if (NbBrackets>0) {
         // More "{" than "}"
         fError = "Missing \"}\"";
         return FormSize(0,0,0);
      }*/ //marthe

      FormSize fs1;
      FormSize fs2;
      FormSize fs3;
      FormSize result;

      // analysis of operators found
      if (OpCloseCurly>-1 && OpCloseCurly<length-1) { // separator } found
         if(!fShow) {
            fs1 = Anal1(spec,text,OpCloseCurly+1);
            fs2 = Anal1(spec,text+OpCloseCurly+1,length-OpCloseCurly-1);
            Savefs(&fs1);
         } else {
            fs1 = Readfs();
            Analyse(x+fs1.Width(),y,spec,text+OpCloseCurly+1,length-OpCloseCurly-1);
            Analyse(x,y,spec,text,OpCloseCurly+1);
         }
         result = fs1+fs2;
      }


      else if (OpPower>-1 && OpUnder>-1) { // ^ and _ found
         min = TMath::Min(OpPower,OpUnder);
         max = TMath::Max(OpPower,OpUnder);
         Double_t xfpos = 0. ; //GetHeight()*spec.size/5.;
         Double_t prop=1, propU=1; // scale factor for #sum & #int
         switch (AbovePlace) {
            case 1 :
               prop = .8 ; propU = 1.5 ; // Int
               break;
            case 2:
               prop = .8 ; propU = 1.5 ; // Sum
               break;
         }
        // propU acts on upper number
        // when increasing propU value, the upper indice position is higher
        // when increasing prop values, the lower indice position is lower

         if (!fShow) {
            Int_t ltext = min ;
            if (min >= 2 && strncmp(&text[min-2],"{}",2)==0) {
               // upper and lower indice before the character
               // like with chemical element
               sprintf(&text[ltext-2],"I ") ;
               ltext-- ;
            }
            fs1 = Anal1(spec,text,ltext);
            fs2 = Anal1(specNewSize,text+min+1,max-min-1);
            fs3 = Anal1(specNewSize,text+max+1,length-max-1);
            Savefs(&fs1);
            Savefs(&fs2);
            Savefs(&fs3);
         } else {
            fs3 = Readfs();
            fs2 = Readfs();
            fs1 = Readfs();
            Double_t pos = 0;
            if (!AbovePlace) {
               Double_t addW = fs1.Width()+xfpos, addH1, addH2;
               if (OpPower<OpUnder) {
                  addH1 = -fs1.Dessus()*(fFactorPos)-fs2.Dessous();
                  addH2 = fs1.Dessous()+fs3.Dessus()*(fFactorPos);
               } else {
                addH1 = fs1.Dessous()+fs2.Dessus()*(fFactorPos);
                addH2 = -fs1.Dessus()*(fFactorPos)-fs3.Dessous();
               }
               Analyse(x+addW,y+addH2,specNewSize,text+max+1,length-max-1);
               Analyse(x+addW,y+addH1,specNewSize,text+min+1,max-min-1);
            } else {
               Double_t addW1, addW2, addH1, addH2;
               Double_t m = TMath::Max(fs1.Width(),TMath::Max(fs2.Width(),fs3.Width()));
               pos = (m-fs1.Width())/2;
               if (OpPower<OpUnder) {
                  addH1 = -fs1.Dessus()*propU-fs2.Dessous();
                  addW1 = (m-fs2.Width())/2;
                  addH2 = fs1.Dessous()*prop+fs3.Dessus();
                  addW2 = (m-fs3.Width())/2;
 //                 if (AbovePlace == 1) addW1 = pos  ;
               } else {
                  addH1 = fs1.Dessous()*prop+fs2.Dessus();
                  addW1 = (m-fs2.Width())/2;
                  addH2 = -fs1.Dessus()*propU-fs3.Dessous();
                  addW2 = (m-fs3.Width())/2;
 //                 if (AbovePlace == 1) addW2 = pos ;
               }

               Analyse(x+addW2,y+addH2,specNewSize,text+max+1,length-max-1);
               Analyse(x+addW1,y+addH1,specNewSize,text+min+1,max-min-1);
            }

            if (min >= 2 && strncmp(&text[min-2],"{}",2)==0) {
               sprintf(&text[min-2],"  ") ;
               Analyse(x+pos,y,spec,text,min-1);
            } else {
               Analyse(x+pos,y,spec,text,min);
            }
         }

         if (!AbovePlace) {
            if (OpPower<OpUnder) {
               result = FormSize(fs1.Width()+xfpos+TMath::Max(fs2.Width(),fs3.Width()),
                                 fs1.Dessus()*fFactorPos+fs2.Height(),
                                 fs1.Dessous()+fs3.Height()-fs3.Dessus()*(1-fFactorPos));
            } else {
               result = FormSize(fs1.Width()+xfpos+TMath::Max(fs2.Width(),fs3.Width()),
                                 fs1.Dessus()*fFactorPos+fs3.Height(),
                                 fs1.Dessous()+fs2.Height()-fs2.Dessus()*(1-fFactorPos));
            }
         } else {
            if (OpPower<OpUnder) {
               result = FormSize(TMath::Max(fs1.Width(),TMath::Max(fs2.Width(),fs3.Width())),
                                 fs1.Dessus()*propU+fs2.Height(),fs1.Dessous()*prop+fs3.Height());
            } else {
               result = FormSize(TMath::Max(fs1.Width(),TMath::Max(fs2.Width(),fs3.Width())),
                                 fs1.Dessus()*propU+fs3.Height(),fs1.Dessous()*prop+fs2.Height());
            }
         }
      }
      else if (OpPower>-1) { // ^ found
        Double_t prop=1;
        Double_t xfpos = 0. ; //GetHeight()*spec.size/5. ;
        switch (AbovePlace) {
           case 1 : //int
              prop = 1.5 ; break ;
           case 2 : // sum
              prop=1.5;  break ;
        }
        // When increasing prop, the upper indice position is higher
        if(!fShow) {
            Int_t ltext = OpPower ;
            if (ltext >= 2 && strncmp(&text[ltext-2],"{}",2)==0) {
               // upper and lower indice before the character
               // like with chemical element
               sprintf(&text[ltext-2],"I ") ;
               ltext-- ;
            }
            fs1 = Anal1(spec,text,ltext);
            fs2 = Anal1(specNewSize,text+OpPower+1,length-OpPower-1);
            Savefs(&fs1);
            Savefs(&fs2);
         } else {
            fs2 = Readfs();
            fs1 = Readfs();
            Int_t pos = 0;
            if (!AbovePlace){
               Analyse(x+fs1.Width()+xfpos,y-fs1.Dessus()*fFactorPos-fs2.Dessous(),specNewSize,text+OpPower+1,length-OpPower-1);
            } else {
               Int_t pos2=0;
               if (fs2.Width()>fs1.Width())
                   pos=Int_t((fs2.Width()-fs1.Width())/2);
               else
                  pos2=Int_t((fs1.Width()-fs2.Width())/2);

               Analyse(x+pos2,y-fs1.Dessus()*prop-fs2.Dessous(),specNewSize,text+OpPower+1,length-OpPower-1);
            }
            if (OpPower >= 2 && strncmp(&text[OpPower-2],"{}",2)==0) {
               sprintf(&text[OpPower-2],"  ") ;
               Analyse(x+pos,y,spec,text,OpPower-1);
            } else {
               Analyse(x+pos,y,spec,text,OpPower);
            }
         }

         if (!AbovePlace)
             result = FormSize(fs1.Width()+xfpos+fs2.Width(),
                               fs1.Dessus()*fFactorPos+fs2.Dessus(),fs1.Dessous());
         else
             result = FormSize(TMath::Max(fs1.Width(),fs2.Width()),fs1.Dessus()*prop+fs2.Height(),fs1.Dessous());

      }
      else if (OpUnder>-1) { // _ found
         Double_t prop = .8; // scale factor for #sum & #frac
         Double_t xfpos = 0.;//GetHeight()*spec.size/5. ;
         Double_t fpos = fFactorPos ;
         // When increasing prop, the lower indice position is lower
         if(!fShow) {
            Int_t ltext = OpUnder ;
            if (ltext >= 2 && strncmp(&text[ltext-2],"{}",2)==0) {
               // upper and lower indice before the character
               // like with chemical element
               sprintf(&text[ltext-2],"I ") ;
               ltext-- ;
            }
            fs1 = Anal1(spec,text,ltext);
            fs2 = Anal1(specNewSize,text+OpUnder+1,length-OpUnder-1);
            Savefs(&fs1);
            Savefs(&fs2);
         } else {
            fs2 = Readfs();
            fs1 = Readfs();
            Int_t pos = 0;
            if (!AbovePlace)
               Analyse(x+fs1.Width()+xfpos,y+fs1.Dessous()+fs2.Dessus()*fpos,specNewSize,text+OpUnder+1,length-OpUnder-1);
            else {
               Int_t pos2=0;
               if (fs2.Width()>fs1.Width())
                  pos=Int_t((fs2.Width()-fs1.Width())/2);
               else
                  pos2=Int_t((fs1.Width()-fs2.Width())/2);

               Analyse(x+pos2,y+fs1.Dessous()*prop+fs2.Dessus(),specNewSize,text+OpUnder+1,length-OpUnder-1);
            }
            if (OpUnder >= 2 && strncmp(&text[OpUnder-2],"{}",2)==0) {
               sprintf(&text[OpUnder-2],"  ") ;
               Analyse(x+pos,y,spec,text,OpUnder-1);
            } else {
               Analyse(x+pos,y,spec,text,OpUnder);
            }
         }
         if (!AbovePlace)
             result = FormSize(fs1.Width()+xfpos+fs2.Width(),fs1.Dessus(),
                               fs1.Dessous()+fs2.Dessous()+fs2.Dessus()*fpos);
         else
            result = FormSize(TMath::Max(fs1.Width(),fs2.Width()),fs1.Dessus(),fs1.Dessous()*prop+fs2.Height());
      }
      else if (OpBox) {
         Double_t square = GetHeight()*spec.size/2;
         if (!fShow) {
            fs1 = Anal1(spec,text+4,length-4);
         } else {
            fs1 = Analyse(x+square,y,spec,text+4,length-4);
            Double_t adjust = GetHeight()*spec.size/20;
            Double_t x1 = x+adjust ;
            Double_t x2 = x-adjust+square ;
            Double_t y1 = y;
            Double_t y2 = y-square+adjust;
            DrawLine(x1,y1,x2,y1,spec);
            DrawLine(x2,y1,x2,y2,spec);
            DrawLine(x2,y2,x1,y2,spec);
            DrawLine(x1,y2,x1,y1,spec);
         }
         result = fs1 + FormSize(square,square,0);
      }
      else if (OpOdot) {
         Double_t square = GetHeight()*spec.size/2;
         if (!fShow) {
            fs1 = Anal1(spec,text+5,length-5);
         } else {
            fs1 = Analyse(x+1.3*square,y,spec,text+5,length-5);
            Double_t adjust = GetHeight()*spec.size/20;
            Double_t r1 = 0.62*square;
            Double_t y1 = y-0.3*square-adjust;
            DrawCircle(x+0.6*square,y1,r1,spec) ;
            DrawCircle(x+0.6*square,y1,r1/100,spec) ;
         }
         result = fs1 + FormSize(square,square,0);
      }
      else if (Operp) {
         Double_t square = GetHeight()*spec.size/1.4;
         if (!fShow) {
            fs1 = Anal1(spec,text+5,length-5);
         } else {
            fs1 = Analyse(x+0.5*square,y,spec,text+5,length-5);
            Double_t x0 = x  + 0.50*square;
            Double_t x1 = x0 - 0.48*square;
            Double_t x2 = x0 + 0.48*square;
            Double_t y1 = y  + 0.6*square;
            Double_t y2 = y1 - 1.3*square;
            DrawLine(x1,y1,x2,y1,spec);
            DrawLine(x0,y1,x0,y2,spec);
         }
         result = fs1;
      }
      else if (Oparallel) {
         Double_t square = GetHeight()*spec.size/1.4;
         if (!fShow) {
            fs1 = Anal1(spec,text+9,length-9);
         } else {
            fs1 = Analyse(x+0.5*square,y,spec,text+9,length-9);
            Double_t x1 = x + 0.15*square;
            Double_t x2 = x + 0.45*square;
            Double_t y1 = y + 0.3*square;
            Double_t y2 = y1- 1.3*square;
            DrawLine(x1,y1,x1,y2,spec);
            DrawLine(x2,y1,x2,y2,spec);
         }
         result = fs1 + FormSize(square,square,0);
      }
      else if (OpGreek>-1) {
         TextSpec_t NewSpec = spec;
         NewSpec.font = 122;
         char letter = 97 + OpGreek;
 //        Double_t yoffset = GetHeight()*spec.size/20.; // Greek letter too low
         Double_t yoffset = 0.; // Greek letter too low
         if (OpGreek>25) letter -= 58;
         if (!fShow) {
            fs1 = Anal1(NewSpec,&letter,1);
            fs2 = Anal1(spec,text+strlen(tab[OpGreek])+1,length-strlen(tab[OpGreek])-1);
            Savefs(&fs1);
         } else {
            fs1 = Readfs();
            Analyse(x+fs1.Width(),y,spec,text+strlen(tab[OpGreek])+1,length-strlen(tab[OpGreek])-1);
            Analyse(x,y-yoffset,NewSpec,&letter,1);
         }
         fs1.add_Dessus(FormSize(0,yoffset,0)) ;
         result = fs1+fs2;
      }

      else if (OpSpec>-1) {
         TextSpec_t NewSpec = spec;
         NewSpec.font = 122;
         char letter = '\243' + OpSpec;
         Double_t props, propi;
         props = 1.8 ; // scale factor for #sum(66)
         propi = 2.3 ; // scale factor for  #int(79)

         if (OpSpec==66 ) {
            NewSpec.size = spec.size*props;
         } else if (OpSpec==79) {
            NewSpec.size = spec.size*propi;
         }
         if (!fShow) {
            fs1 = Anal1(NewSpec,&letter,1);
            if (OpSpec == 79 || OpSpec == 66)
                 fs1 = FormSize(fs1.Width(),fs1.Dessus()*0.4,fs1.Dessus()*0.40);

            fs2 = Anal1(spec,text+strlen(tab2[OpSpec])+1,length-strlen(tab2[OpSpec])-1);
            Savefs(&fs1);
         } else {
            fs1 = Readfs();
            Analyse(x+fs1.Width(),y,spec,text+strlen(tab2[OpSpec])+1,length-strlen(tab2[OpSpec])-1);
            if (OpSpec!=66 && OpSpec!=79)
               Analyse(x,y,NewSpec,&letter,1);
            else {
                  Analyse(x,y+fs1.Dessous()/2.,NewSpec,&letter,1);
            }
         }
         result = fs1+fs2;
      }
      else if (OpAbove>-1) {
         if (!fShow) {
            fs1 = Anal1(spec,text+strlen(tab3[OpAbove])+1,length-strlen(tab3[OpAbove])-1);
            Savefs(&fs1);
         } else {
            fs1 = Readfs();
            Analyse(x,y,spec,text+strlen(tab3[OpAbove])+1,length-strlen(tab3[OpAbove])-1);
//            Double_t sub = GetHeight()*spec.size/12;
            Double_t sub = GetHeight()*spec.size/14;
            Double_t x1 , y1 , x2, y2, x3, x4;
            switch(OpAbove) {
            case 0: // bar
               Double_t ypos  ;
               ypos = y-fs1.Dessus()-sub ;//-GetHeight()*spec.size/4. ;
               DrawLine(x,ypos,x+fs1.Width(),ypos,spec);
               break;
            case 1: // vec
               Double_t y0 ;
               y0 = y-sub-fs1.Dessus() ;
               y1 = y0-GetHeight()*spec.size/8 ;
               x1 = x+fs1.Width() ;
               DrawLine(x,y1,x1,y1,spec);
               DrawLine(x1,y1,x1-GetHeight()*spec.size/4,y0-GetHeight()*spec.size/4,spec);
               DrawLine(x1,y1,x1-GetHeight()*spec.size/4,y0,spec);
               break;
            case 2: // dot
               x1 = x+fs1.Width()/2-3*sub/4 ;
               x2 = x+fs1.Width()/2+3*sub/4 ;
               y1 = y-sub-fs1.Dessus() ;
               DrawLine(x1,y1,x2,y1,spec);
               break;
            case 3: // hat
               x2 = x+fs1.Width()/2 ;
               y1 = y +sub -fs1.Dessus() ;
               y2 = y1-2*sub;
               x1 = x2-fs1.Width()/3 ;
               x3 = x2+fs1.Width()/3 ;
               DrawLine(x1,y1,x2,y2,spec);
               DrawLine(x2,y2,x3,y1,spec);
               break;
            case 4: // ddot
               x1 = x+fs1.Width()/2-9*sub/4 ;
               x2 = x+fs1.Width()/2-3*sub/4 ;
               x3 = x+fs1.Width()/2+9*sub/4 ;
               x4 = x+fs1.Width()/2+3*sub/4 ;
               y1 = y-sub-fs1.Dessus() ;
               DrawLine(x1,y1,x2,y1,spec);
               DrawLine(x3,y1,x4,y1,spec);
               break;
            case 5: // acute
               x1 = x+fs1.Width()/2-0.5*sub;
               y1 = y +sub -fs1.Dessus() ;
               x2 = x1 +2*sub;
               y2 = y1 -2*sub;
               DrawLine(x1,y1,x2,y2,spec);
               break;
            case 6: // grave
               x1 = x+fs1.Width()/2-sub;
               y1 = y-sub-fs1.Dessus() ;
               x2 = x1 +2*sub;
               y2 = y1 +2*sub;
               DrawLine(x1,y1,x2,y2,spec);
               break;
            case 7: // check
               x1 = x+fs1.Width()/2 ;
               x2 = x1 -2*sub ;
               x3 = x1 +2*sub ;
               y1 = y-sub-fs1.Dessus() ;
               DrawLine(x2,y-3*sub-fs1.Dessus(),x1,y1,spec);
               DrawLine(x3,y-3*sub-fs1.Dessus(),x1,y1,spec);
               break;
           }
         }
         Double_t div = 3;
         if (OpAbove==1) div=4;
         result = FormSize(fs1.Width(),fs1.Dessus()+GetHeight()*spec.size/div,fs1.Dessous());

      }
      else if (OpSquareBracket) { // operator #[]{arg}
         Double_t l = GetHeight()*spec.size/4;
         Double_t l2 = l/2 ;
         if (!fShow) {
            fs1 = Anal1(spec,text+3,length-3);
            Savefs(&fs1);
         } else {
            fs1 = Readfs();
            Analyse(x+l2+l,y,spec,text+3,length-3);
            DrawLine(x+l2,y-fs1.Dessus(),x+l2,y+fs1.Dessous(),spec);
            DrawLine(x+l2,y-fs1.Dessus(),x+l2+l,y-fs1.Dessus(),spec);
            DrawLine(x+l2,y+fs1.Dessous(),x+l2+l,y+fs1.Dessous(),spec);
            DrawLine(x+l2+fs1.Width()+2*l,y-fs1.Dessus(),x+l2+fs1.Width()+2*l,y+fs1.Dessous(),spec);
            DrawLine(x+l2+fs1.Width()+2*l,y-fs1.Dessus(),x+l2+fs1.Width()+l,y-fs1.Dessus(),spec);
            DrawLine(x+l2+fs1.Width()+2*l,y+fs1.Dessous(),x+l2+fs1.Width()+l,y+fs1.Dessous(),spec);
         }
         result = FormSize(fs1.Width()+3*l,fs1.Dessus(),fs1.Dessous());
      }
      else if (OpParen) {  // operator #(){arg}
         Double_t l = GetHeight()*spec.size/4;
         Double_t radius2,radius1 , dw, l2 = l/2 ;
         Double_t angle = 35 ;
         if (!fShow) {
            fs1 = Anal1(spec,text+3,length-3);
            Savefs(&fs1);
            radius2 = fs1.Height() ;
            radius1 = radius2  * 2 / 3;
            dw = radius1*(1 - TMath::Cos(kPI*angle/180)) ;
         } else {
            fs1 = Readfs();
            radius2 = fs1.Height() ;
            radius1 = radius2  * 2 / 3;
            dw = radius1*(1 - TMath::Cos(kPI*angle/180)) ;
            Double_t x1 = x+l2+radius1 ;
            Double_t x2 = x+5*l2+2*dw+fs1.Width()-radius1 ;
            Double_t y1 = y - (fs1.Dessus() - fs1.Dessous())/2. ;
            DrawParenthesis(x1,y1,radius1,radius2,180-angle,180+angle,spec) ;
            DrawParenthesis(x2,y1,radius1,radius2,360-angle,360+angle,spec) ;
            Analyse(x+3*l2+dw,y,spec,text+3,length-3);
         }
        // result = FormSize(fs1.Width()+3*l,fs1.Dessus(),fs1.Dessous());
         result = FormSize(fs1.Width()+3*l+2*dw,fs1.Dessus(),fs1.Dessous());
      }
      else if (OpAbs) {  // operator #||{arg}
         Double_t l = GetHeight()*spec.size/4;
         Double_t l2 = l/2 ;
         if (!fShow) {
            fs1 = Anal1(spec,text+3,length-3);
            Savefs(&fs1);
         } else {
            fs1 = Readfs();
            Analyse(x+l2+l,y,spec,text+3,length-3);
            DrawLine(x+l2,y-fs1.Dessus(),x+l2,y+fs1.Dessous(),spec);
            DrawLine(x+l2+fs1.Width()+2*l,y-fs1.Dessus(),x+l2+fs1.Width()+2*l,y+fs1.Dessous(),spec);
         }
         result = FormSize(fs1.Width()+3*l,fs1.Dessus(),fs1.Dessous());
      }
      else if (OpBigCurly) { // big curly bracket  #{}{arg}
         Double_t l = GetHeight()*spec.size/4;
         Double_t l2 = l/2 ;
         Double_t l8 , ltip;

         if (!fShow) {
            fs1 = Anal1(spec,text+3,length-3);
            l8 = fs1.Height()/8 ;
            ltip = TMath::Min(l8,l) ;
            l = ltip ;
            Savefs(&fs1);
         } else {
            fs1 = Readfs();
            Double_t y2 = y + (fs1.Dessous()-fs1.Dessus())/2 ;
            l8 = fs1.Height()/8 ;
            ltip = TMath::Min(l8,l) ;
            l = ltip ;
            Analyse(x+l+ltip+l2,y,spec,text+3,length-3);
            // Draw open curly bracket
            // Vertical lines
            DrawLine(x+l2+ltip,y-fs1.Dessus(),x+l2+ltip,y2-ltip,spec);
            DrawLine(x+l2+ltip,y2+ltip,x+l2+ltip,y+fs1.Dessous(),spec);
            // top and bottom lines
            DrawLine(x+l2+ltip,y-fs1.Dessus(),x+l2+ltip+l,y-fs1.Dessus(),spec);
            DrawLine(x+l2+ltip,y+fs1.Dessous(),x+l2+ltip+l,y+fs1.Dessous(),spec);
            // < sign
            DrawLine(x+l2,y2,x+l2+ltip,y2-ltip,spec);
            DrawLine(x+l2,y2,x+l2+ltip,y2+ltip,spec);

            // Draw close curly bracket
            // vertical lines
            DrawLine(x+l2+ltip+fs1.Width()+2*l,y-fs1.Dessus(),x+l2+ltip+fs1.Width()+2*l,y2-ltip,spec);
            DrawLine(x+l2+ltip+fs1.Width()+2*l,y2+ltip,x+l2+ltip+fs1.Width()+2*l,y+fs1.Dessous(),spec);
            // Top and bottom lines
            DrawLine(x+l2+fs1.Width()+l+ltip,y-fs1.Dessus(),x+l2+ltip+fs1.Width()+2*l,y-fs1.Dessus(),spec);
            DrawLine(x+l2+fs1.Width()+l+ltip,y+fs1.Dessous(),x+l2+ltip+fs1.Width()+2*l,y+fs1.Dessous(),spec);
            // > sign
            DrawLine(x+l2+ltip+2*l+fs1.Width(),y2-ltip,x+l2+2*l+2*ltip+fs1.Width(),y2,spec);
            DrawLine(x+l2+ltip+2*l+fs1.Width(),y2+ltip,x+l2+2*l+2*ltip+fs1.Width(),y2,spec);
         }
         result = FormSize(fs1.Width()+3*l+2*ltip,fs1.Dessus(),fs1.Dessous()) ;;
      }
      else if (OpFrac>-1) { // \frac found
         if (OpCurlyCurly==-1) { // }{ not found
            // arguments missing for \frac
            fError = "Missing denominator for #frac";
            return FormSize(0,0,0);
         }
         Double_t height = GetHeight()*spec.size/8;
         if (!fShow) {
            fs1 = Anal1(spec,text+OpFrac+6,OpCurlyCurly-OpFrac-6);
            fs2 = Anal1(spec,text+OpCurlyCurly+2,length-OpCurlyCurly-3);
            Savefs(&fs1);
            Savefs(&fs2);
         } else {
            fs2 = Readfs();
            fs1 = Readfs();
            Double_t addW1,addW2;
            if (fs1.Width()<fs2.Width()) {
               addW1 = (fs2.Width()-fs1.Width())/2;
               addW2 = 0;
            } else {
               addW1 = 0;
               addW2 = (fs1.Width()-fs2.Width())/2;
            }
            Analyse(x+addW2,y+fs2.Dessus()-height,spec,text+OpCurlyCurly+2,length-OpCurlyCurly-3);  // denominator
            Analyse(x+addW1,y-fs1.Dessous()-3*height,spec,text+OpFrac+6,OpCurlyCurly-OpFrac-6); //numerator

            DrawLine(x,y-2*height,x+TMath::Max(fs1.Width(),fs2.Width()),y-2*height,spec);
         }

         result = FormSize(TMath::Max(fs1.Width(),fs2.Width()),fs1.Height()+3*height,fs2.Height()-height);

      }
      else if (OpSqrt>-1) { // \sqrt found
         if (!fShow) {
            if (OpSquareCurly>-1) {
               // power nth  #sqrt[n]{arg}
               fs1 = Anal1(specNewSize,text+OpSqrt+6,OpSquareCurly-OpSqrt-6);
               fs2 = Anal1(spec,text+OpSquareCurly+1,length-OpSquareCurly-1);
               Savefs(&fs1);
               Savefs(&fs2);
               result = FormSize(fs2.Width()+ GetHeight()*spec.size/10+TMath::Max(GetHeight()*spec.size/2,(Double_t)fs1.Width()),
                                 fs2.Dessus()+fs1.Height()+GetHeight()*spec.size/4,fs2.Dessous());
            } else {
               fs1 = Anal1(spec,text+OpSqrt+5,length-OpSqrt-5);
               Savefs(&fs1);
               result = FormSize(fs1.Width()+GetHeight()*spec.size/2,fs1.Dessus()+GetHeight()*spec.size/4,fs1.Dessous());
            }
         } else {
            if (OpSquareCurly>-1) { // ]{
               fs2 = Readfs();
               fs1 = Readfs();
               Double_t pas = TMath::Max(GetHeight()*spec.size/2,(Double_t)fs1.Width());
               Double_t pas2 = pas + GetHeight()*spec.size/10;
               Double_t y1 = y-fs2.Dessus() ;
               Double_t y2 = y+fs2.Dessous() ;
               Double_t y3 = y1-GetHeight()*spec.size/4;
               Analyse(x+pas2,y,spec,text+OpSquareCurly+1,length-OpSquareCurly-1);
               Analyse(x,y-fs2.Dessus()-fs1.Dessous(),specNewSize,text+OpSqrt+6,OpSquareCurly-OpSqrt-6); // indice
               DrawLine(x,y1,x+pas,y2,spec);
               DrawLine(x+pas,y2,x+pas,y3,spec);
               DrawLine(x+pas,y3,x+pas2+fs2.Width(),y3,spec);
            } else {
               fs1 = Readfs();
               Double_t x1 = x+GetHeight()*spec.size*2/5 ;
               Double_t x2 = x+GetHeight()*spec.size/2+fs1.Width() ;
               Double_t y1 = y-fs1.Dessus() ;
               Double_t y2 = y+fs1.Dessous() ;
               Double_t y3 = y1-GetHeight()*spec.size/4;

               Analyse(x+GetHeight()*spec.size/2,y,spec,text+OpSqrt+6,length-OpSqrt-7);
               DrawLine(x,y1,x1,y2,spec);
               DrawLine(x1,y2,x1,y3,spec);
               DrawLine(x1,y3,x2,y3,spec);

            }
         }
      }
      else if (OpColor>-1) { // \color found
         if (OpSquareCurly==-1) {
            // color number is not specified
            fError = "Missing color number. Syntax is #color[(Int_t)nb]{ ... }";
            return FormSize(0,0,0);
         }
         TextSpec_t NewSpec = spec;
         Char_t *nb = new Char_t[OpSquareCurly-OpColor-7];
         strncpy(nb,text+OpColor+7,OpSquareCurly-OpColor-7);
         if (sscanf(nb,"%d",&NewSpec.color) < 1) {
            delete[] nb;
            // color number is invalid
            fError = "Invalid color number. Syntax is #color[(Int_t)nb]{ ... }";
            return FormSize(0,0,0);
         }
         delete[] nb;
         if (!fShow) {
            result = Anal1(NewSpec,text+OpSquareCurly+1,length-OpSquareCurly-1);
         } else {
            Analyse(x,y,NewSpec,text+OpSquareCurly+1,length-OpSquareCurly-1);
         }
      }
      else if (OpFont>-1) { // \font found
         if (OpSquareCurly==-1) {
            // font number is not specified
            fError = "Missing font number. Syntax is #font[nb]{ ... }";
            return FormSize(0,0,0);
         }
         TextSpec_t NewSpec = spec;
         Char_t *nb = new Char_t[OpSquareCurly-OpFont-6];
         strncpy(nb,text+OpFont+6,OpSquareCurly-OpFont-6);
         if (sscanf(nb,"%d",&NewSpec.font) < 1) {
            delete[] nb;
            // font number is invalid
            fError = "Invalid font number. Syntax is #font[(Int_t)nb]{ ... }";
            return FormSize(0,0,0);
         }
         delete[] nb;
         if (!fShow) {
            result = Anal1(NewSpec,text+OpSquareCurly+1,length-OpSquareCurly-1);
         } else {
            Analyse(x,y,NewSpec,text+OpSquareCurly+1,length-OpSquareCurly-1);
         }
      } else { // no operators found, it is a character string
         SetTextSize(spec.size);
         SetTextAngle(spec.angle);
         SetTextColor(spec.color);
         SetTextFont(spec.font);
         SetTextAlign(11);
         TAttText::Modify();
         UInt_t w=0,h=0;

         Int_t leng = strlen(text) ;

         quote1 = quote2 = kFALSE ;
         Char_t *p ;
         for (i=0 ; i<leng ; i++) {
            switch (text[i]) {
               case '\'' : quote1 = !quote1 ; break ; // single quote symbol not correctly interpreted when PostScript
               case '"'  : quote2 = !quote2 ;  break ;
            }
            //if (quote1 || quote2) continue ;
            if (text[i] == '@') {  // @ symbol not correctly interpreted when PostScript
               p = &text[i] ;
               if ( *(p+1) == '{' || *(p+1) == '}' || *(p+1) == '[' || *(p+1) == ']') {
                  while (*p != 0) {
                     *p = *(p+1) ; p++ ;
                  }
                  leng-- ;
               }
            }
         }
         text[leng] = 0 ;

/* // skip the @ character in all cases
         Char_t *p = text ;
         while (p) {
            p = strchr(p,'@');
            if (p) {
               while (*p != 0) {
                  *p = *(p+1) ; p++ ;
               }
               leng--; text[leng] = 0 ;
            }
         }
*/
         GetTextExtent(w,h,text);
         Double_t hy    = h;
         Double_t width = w;

         fs1 = FormSize(width,hy,0);
         if (fShow) {
            // paint the Latex sub-expression per sub-expression
            Double_t Xorigin = (Double_t)gPad->XtoAbsPixel(fX);
            Double_t Yorigin = (Double_t)gPad->YtoAbsPixel(fY);
            Double_t angle   = kPI*spec.angle/180.;
            Double_t X = gPad->AbsPixeltoX(Int_t((x-Xorigin)*TMath::Cos(angle)+(y-Yorigin)*TMath::Sin(angle)+Xorigin));
            Double_t Y = gPad->AbsPixeltoY(Int_t((x-Xorigin)*TMath::Sin(-angle)+(y-Yorigin)*TMath::Cos(angle)+Yorigin));
            gPad->PaintText(X,Y,text);
         }

         result = fs1;
      }

      delete[] text;

      return result;
}

//______________________________________________________________________________
TLatex *TLatex::DrawLatex(Double_t x, Double_t y, const char *text)
{
// Make a copy of this object with the new parameters
// And copy object attributes

   TLatex *newtext = new TLatex(x, y, text);
   TAttText::Copy(*newtext);
   newtext->SetBit(kCanDelete);
   if (TestBit(kTextNDC)) newtext->SetNDC();
   newtext->AppendPad();
   return newtext;
}

//______________________________________________________________________________
void TLatex::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2, TextSpec_t spec)
{
// Draw a line in a Latex formula
      Double_t sinang  = TMath::Sin(spec.angle/180*kPI);
      Double_t cosang  = TMath::Cos(spec.angle/180*kPI);
      Double_t Xorigin = (Double_t)gPad->XtoAbsPixel(fX);
      Double_t Yorigin = (Double_t)gPad->YtoAbsPixel(fY);
      Double_t X  = gPad->AbsPixeltoX(Int_t((x1-Xorigin)*cosang+(y1-Yorigin)*sinang+Xorigin));
      Double_t Y  = gPad->AbsPixeltoY(Int_t((x1-Xorigin)*-sinang+(y1-Yorigin)*cosang+Yorigin));

      Double_t X2 = gPad->AbsPixeltoX(Int_t((x2-Xorigin)*cosang+(y2-Yorigin)*sinang+Xorigin));
      Double_t Y2 = gPad->AbsPixeltoY(Int_t((x2-Xorigin)*-sinang+(y2-Yorigin)*cosang+Yorigin));

//      Short_t lw = Short_t(GetHeight()*spec.size/8);
//      SetLineWidth(lw);
      SetLineColor(spec.color);
      TAttLine::Modify();
      gPad->PaintLine(X,Y,X2,Y2);
}

//______________________________________________________________________________
void TLatex::DrawCircle(Double_t x1, Double_t y1, Double_t r, TextSpec_t spec )
{
// Draw an arc of ellipse in a Latex formula (right or left parenthesis)

   if (r < 1) r = 1;
   Double_t sinang  = TMath::Sin(spec.angle/180*kPI);
   Double_t cosang  = TMath::Cos(spec.angle/180*kPI);
   Double_t Xorigin = (Double_t)gPad->XtoAbsPixel(fX);
   Double_t Yorigin = (Double_t)gPad->YtoAbsPixel(fY);

   const Int_t np = 40;
   Double_t dphi = 2*kPI/np;
   Double_t x[np+3], y[np+3];
   Double_t angle,dx,dy;

   SetLineColor(spec.color);
   TAttLine::Modify();  //Change line attributes only if necessary

   for (Int_t i=0;i<=np;i++) {
      angle = Double_t(i)*dphi;
      dx    = r*TMath::Cos(angle) +x1 -Xorigin;
      dy    = r*TMath::Sin(angle) +y1 -Yorigin;
      x[i]  = gPad->AbsPixeltoX(Int_t( dx*cosang+ dy*sinang +Xorigin));
      y[i]  = gPad->AbsPixeltoY(Int_t(-dx*sinang+ dy*cosang +Yorigin));
   }
   gPad->PaintPolyLine(np+1,x,y);

}

//______________________________________________________________________________
void TLatex::DrawParenthesis(Double_t x1, Double_t y1, Double_t r1, Double_t r2,
                     Double_t  phimin, Double_t  phimax, TextSpec_t spec )
{
// Draw an arc of ellipse in a Latex formula (right or left parenthesis)

   if (r1 < 1) r1 = 1;
   if (r2 < 1) r2 = 1;
   Double_t sinang  = TMath::Sin(spec.angle/180*kPI);
   Double_t cosang  = TMath::Cos(spec.angle/180*kPI);
   Double_t Xorigin = (Double_t)gPad->XtoAbsPixel(fX);
   Double_t Yorigin = (Double_t)gPad->YtoAbsPixel(fY);

   const Int_t np = 40;
   Double_t dphi = (phimax-phimin)*kPI/(180*np);
   Double_t x[np+3], y[np+3];
   Double_t angle,dx,dy ;

   SetLineColor(spec.color);
   TAttLine::Modify();  //Change line attributes only if necessary

   for (Int_t i=0;i<=np;i++) {
      angle = phimin*kPI/180 + Double_t(i)*dphi;
      dx    = r1*TMath::Cos(angle) +x1 -Xorigin;
      dy    = r2*TMath::Sin(angle) +y1 -Yorigin;
      x[i]  = fX+gPad->AbsPixeltoX(Int_t( dx*cosang+dy*sinang));
      y[i]  = fY+gPad->AbsPixeltoY(Int_t(-dx*sinang+dy*cosang));
   }
   gPad->PaintPolyLine(np+1,x,y);

}

//______________________________________________________________________________
void TLatex::Paint(Option_t *)
{
// Paint
  if (TestBit(kTextNDC)) {
     Double_t xsave = fX;
     Double_t ysave = fY;
     fX = gPad->GetX1() + xsave*(gPad->GetX2() - gPad->GetX1());
     fY = gPad->GetY1() + ysave*(gPad->GetY2() - gPad->GetY1());
     PaintLatex(fX,fY,GetTextAngle(),GetTextSize(),GetTitle());
     fX = xsave;
     fY = ysave;
  } else {
     PaintLatex(fX,fY,GetTextAngle(),GetTextSize(),GetTitle());
  }
}

//______________________________________________________________________________
void TLatex::PaintLatex(Double_t x, Double_t y, Double_t angle, Double_t size, const Char_t *text1)
{
// Main drawing function


      TAttText::Modify();  //Change text attributes only if necessary

       // do not use Latex if font is low precision
      if (fTextFont%10 < 2) {
         gPad->PaintText(x,y,text1);
         return;
      }

      Double_t saveSize = size;
      Int_t saveFont = fTextFont;
      if (fTextFont%10 > 2) {
         if (gPad->GetWw() < gPad->GetWh())
            size = size/gPad->GetWw();
         else
            size = size/gPad->GetWh();
         SetTextFont(10*(saveFont/10) + 2);
      }
      if (gVirtualPS) gVirtualPS->SetBit(kLatex);

      TString newText = text1;

      if( newText.Length() == 0) return;

      fError = 0 ;
      if (CheckLatexSyntax(newText)) {
         cout<<"\n*ERROR<TLatex>: "<<fError<<endl;
         cout<<"==> "<<text1<<endl;
         return ;
      }
      fError = 0 ;

      Int_t length = newText.Length() ;
      const Char_t *text = newText.Data() ;

      fX=x;
      fY=y;
      x = gPad->XtoAbsPixel(x);
      y = gPad->YtoAbsPixel(y);
      fShow = kFALSE ;
      FormSize fs = FirstParse(angle,size,text);

      fOriginSize = size;

      //get current line attributes
      Short_t lineW = GetLineWidth();
      Int_t lineC = GetLineColor();

      TextSpec_t spec;
      spec.angle = angle;
      spec.size  = size;
      spec.color = GetTextColor();
      spec.font  = GetTextFont();
      Short_t halign = fTextAlign/10;
      Short_t valign = fTextAlign - 10*halign;
      TextSpec_t NewSpec = spec;
      if (fError != 0) {
         cout<<"*ERROR<TLatex>: "<<fError<<endl;
         cout<<"==> "<<text<<endl;
      } else {
         fShow = kTRUE;
         Double_t mul = 1;
         NewSpec.size = mul*size;

         switch (valign) {
            case 0: y -= fs.Dessous()*mul; break;
            case 1: break;
            case 2: y += (fs.Dessus()-fs.Dessous())*mul/2.5; break;
            case 3: y += fs.Dessus()*mul;  break;
         }
         switch (halign) {
            case 2: x -= fs.Width()*mul/2  ; break;
            case 3: x -= fs.Width()*mul    ;   break;
         }
         Analyse(x,y,NewSpec,text,length);
      }

      SetTextSize(saveSize);
      SetTextAngle(angle);
      SetTextFont(saveFont);
      SetTextColor(spec.color);
      SetTextAlign(valign+10*halign);
      SetLineWidth(lineW);
      SetLineColor(lineC);
      delete[] fTabSize;

      if (gVirtualPS) gVirtualPS->ResetBit(kLatex);
}

//______________________________________________________________________________
Int_t TLatex::CheckLatexSyntax(TString &text)
{
   // Check if the Latex syntax is correct

   const Char_t *kWord1[] = {"{}^{","{}_{","^{","_{","#color{","#font{","#sqrt{","#[]{","#{}{","#||{",
                       "#bar{","#vec{","#dot{","#hat{","#ddot{","#acute{","#grave{","#check{",
                       "\\color{","\\font{","\\sqrt{","\\[]{","\\{}{","\\||{","#(){","\\(){",
                       "\\bar{","\\vec{","\\dot{","\\hat{","\\ddot{","\\acute{","\\grave{","\\check{"}; // check for }
   const Char_t *kWord2[] = {"#color[","#font[","#sqrt[","\\color[","\\font[","\\sqrt["}; // check for ]{ + }
   const Char_t *kWord3[] = {"#frac{","\\frac{"} ; // check for }{ then }
   const Char_t *kLeft1[] = {"#left[","\\left[","#left{","\\left{","#left|","\\left|","#left(","\\left("} ;
   const Char_t *kLeft2[] = {"#[]{","#[]{","#{}{","#{}{","#||{","#||{","#(){","#(){"} ;
   const Char_t *kRight[] = {"#right]","\\right]","#right}","\\right}","#right|","\\right|","#right)","\\right)"} ;
   Int_t lkWord1[] = {4,4,2,2,7,6,6,4,4,4,
                      5,5,5,5,6,7,7,7,
                      7,6,6,4,4,4,4,4,
                      5,5,5,5,6,7,7,7} ;
   Int_t lkWord2[] = {7,6,6,7,6,6} ;
   Int_t lkWord3[] = {6,6} ;
   Int_t NkWord1 = 34, NkWord2 = 6, NkWord3 = 2 ;
   Int_t nLeft1 , nRight , nOfLeft, nOfRight;
   Int_t lLeft1 = 6 ;
   Int_t lLeft2 = 4 ;
   Int_t lRight = 7 ;
   nLeft1  = nRight   = 8 ;
   nOfLeft = nOfRight = 0 ;

   Int_t i,k ;
   Char_t buf[7] ;
   Bool_t opFound ;
   Int_t  opFrac = 0;
   Int_t length = text.Length() ;

   Int_t nOfCurlyBracket, nOfKW1, nOfKW2, nOfKW3, nOfSquareCurly, nOfCurlyCurly ;
   Int_t nOfExtraCurly = 0 , nOfExtraSquare = 0;
   Int_t nOfSquareBracket = 0 ;
   Int_t error = 0  ;
   Bool_t quote1 = kFALSE , quote2 = kFALSE;

   // first find and replace all occurences of "kLeft1" keyword by "kLeft2" keyword,
   // and all occurences of "kRight" keyword by "}".
   i = 0 ;
   while (i < length) {
      strncpy(buf,&text[i],TMath::Min(7,length-i));
      opFound = kFALSE ;
      for (k = 0 ; k < nLeft1 ; k++) {
          if (strncmp(buf,kLeft1[k],lLeft1)==0) {
               nOfLeft++ ;
               i+=lLeft1 ;
               opFound = kTRUE ;
               break ;
           }
       }
       if (opFound) continue ;

       for(k=0;k<nRight;k++) {
          if (strncmp(buf,kRight[k],lRight)==0) {
             nOfRight++ ;
             i+=lRight ;
             opFound = kTRUE ;
             break ;
          }
       }
       if (!opFound) i++ ;
   }
   if (nOfLeft != nOfRight) {
      printf(" nOfLeft = %d, nOfRight = %d\n",nOfLeft,nOfRight) ;
      error = 1 ;
      fError = "Operators \"#left\" and \"#right\" don't match !" ;
      goto ERROR_END ;
   }

   for (k = 0 ; k < nLeft1 ; k++) {
      text.ReplaceAll(kLeft1[k],lLeft1,kLeft2[k],lLeft2) ;
   }
   for (k = 0 ; k < nRight ; k++) {
      text.ReplaceAll(kRight[k],lRight,"}",1) ;
   }
   length = text.Length() ;

   i = nOfCurlyBracket = nOfKW1 = nOfKW2 = nOfKW3 = nOfSquareCurly = nOfCurlyCurly =0 ;
   while (i< length){
         switch (text[i]) {
           case '"' : quote1 = !quote1 ; break ;
           case '\'': quote2 = !quote2 ; break ;
         }
         //if (quote1 || quote2) {
         //   i++;
         //   continue ;
         //}
         strncpy(buf,&text[i],TMath::Min(7,length-i));
         opFound = kFALSE ;

         for(k=0;k<NkWord1;k++) {
            if (strncmp(buf,kWord1[k],lkWord1[k])==0) {
               nOfKW1++ ;
               i+=lkWord1[k] ;
               opFound = kTRUE ;
               nOfCurlyBracket++ ;
               break ;
            }
         }
         if (opFound) continue ;

         for(k=0;k<NkWord2;k++) {
            if (strncmp(buf,kWord2[k],lkWord2[k])==0) {
               nOfKW2++ ;
               i+=lkWord2[k] ;
               opFound = kTRUE ;
               nOfSquareBracket++;
               break ;
            }
         }
         if (opFound) continue ;

         for(k=0;k<NkWord3;k++) {
            if (strncmp(buf,kWord3[k],lkWord3[k])==0) {
               nOfKW3++ ;
               i+=lkWord3[k] ;
               opFound = kTRUE ;
               opFrac++ ;
               nOfCurlyBracket++ ;
               break ;
            }
         }
         if (opFound) continue ;
         if (strncmp(buf,"}{",2) == 0 && opFrac) {
               opFrac-- ;
               nOfCurlyCurly++ ;
               i+= 2;
         }
         else if (strncmp(buf,"]{",2) == 0 && nOfSquareBracket) {
               nOfSquareCurly++ ;
               i+= 2 ;
               nOfCurlyBracket++ ;
               nOfSquareBracket-- ;
         }
         else if (strncmp(buf,"@{",2) == 0 || strncmp(buf,"@}",2) == 0) {
               i+= 2 ;
         }
         else if (strncmp(buf,"@[",2) == 0 || strncmp(buf,"@]",2) == 0) {
               i+= 2 ;
         }
         else if (text[i] == ']' ) {  // not belonging to a key word, add @ in front
               text.Insert(i,"@") ;
               length++ ;
               i+=2 ;
               nOfExtraSquare-- ;
         }
         else if (text[i] == '[' ) {  // not belonging to a key word, add @ in front
               text.Insert(i,"@") ;
               length++ ;
               i+=2 ;
               nOfExtraSquare++ ;
         }
         else if (text[i] == '{' ) {  // not belonging to a key word, add @ in front
               text.Insert(i,"@") ;
               length++ ;
               i+=2 ;
               nOfExtraCurly++ ;
         }
         else if (text[i] == '}' ) {
            if ( nOfCurlyBracket) {
               nOfCurlyBracket-- ;
               i++ ;
            } else  { // extra }, add @ in front
               text.Insert(i,"@") ;
               length++ ;
               i+=2 ;
               nOfExtraCurly-- ;
            }
         } else {
            i++ ;
            buf[1] = 0 ;
         }
   }

   if (nOfKW2 != nOfSquareCurly) {
      error = 1 ;
      fError = "Invalid number of \"]{\"" ;
   }
   else if (nOfKW3 != nOfCurlyCurly) {
      error = 1 ;
      fError = "Error in syntax of  \"#frac\"" ;
   }
   else if (nOfCurlyBracket  < 0) {
      error = 1 ;
      fError = "Missing \"{\"" ;
   }
   else if (nOfCurlyBracket  > 0) {
      error = 1 ;
      fError = "Missing \"}\"" ;
   }
   else if (nOfSquareBracket  < 0) {
      error  = 1 ;
      fError = "Missing \"[\"" ;
   }
   else if (nOfSquareBracket  > 0) {
      error = 1 ;
      fError = "Missing \"]\"" ;
   }

   ERROR_END:
   return error ;
 }

//______________________________________________________________________________
FormSize TLatex::FirstParse(Double_t angle, Double_t size, const Char_t *text) {
// first parsing of the analyse sequence
      fError   = 0;
      fTabMax  = 100;
      fTabSize = new FormSize_t[fTabMax];
      // we assume less than 100 parts in one formula
      // we will reallocate if necessary.
      fPos        = 0;
      fShow       = kFALSE;
      fOriginSize = size;

      //get current line attributes
      Short_t lineW = GetLineWidth();
      Int_t lineC = GetLineColor();

      TextSpec_t spec;
      spec.angle = angle;
      spec.size  = size;
      spec.color = GetTextColor();
      spec.font  = GetTextFont();
      Short_t halign = fTextAlign/10;
      Short_t valign = fTextAlign - 10*halign;


      FormSize fs = Anal1(spec,text,strlen(text));

      SetTextSize(size);
      SetTextAngle(angle);
      SetTextFont(spec.font);
      SetTextColor(spec.color);
      SetTextAlign(valign+10*halign);
      SetLineWidth(lineW);
      SetLineColor(lineC);
      return fs;
}

//______________________________________________________________________________
Double_t TLatex::GetHeight() const
{
// return height of current pad in pixels

   if (gPad->GetWw() < gPad->GetWh())
      return gPad->GetAbsWNDC()*Double_t(gPad->GetWw());
   else
      return gPad->GetAbsHNDC()*Double_t(gPad->GetWh());
}

//______________________________________________________________________________
Double_t TLatex::GetXsize()
{
// return size of the formula along X in pad coordinates
      if (!gPad) return 0;
      TString newText = GetTitle();
      if( newText.Length() == 0) return 0;
      fError = 0 ;
      if (CheckLatexSyntax(newText)) {
         cout<<"\n*ERROR<TLatex>: "<<fError<<endl;
         cout<<"==> "<<GetTitle()<<endl;
         return 0;
      }
      fError = 0 ;

      const Char_t *text = newText.Data() ;
      FormSize fs = FirstParse(0,GetTextSize(),text);
      delete[] fTabSize;
      return TMath::Abs(gPad->AbsPixeltoX(Int_t(fs.Width())) - gPad->AbsPixeltoX(0));
}

//______________________________________________________________________________
Double_t TLatex::GetYsize()
{
// return size of the formula along Y in pad coordinates
      if (!gPad) return 0;
      TString newText = GetTitle();
      if( newText.Length() == 0) return 0;
      fError = 0 ;
      if (CheckLatexSyntax(newText)) {
         cout<<"\n*ERROR<TLatex>: "<<fError<<endl;
         cout<<"==> "<<GetTitle()<<endl;
         return 0;
      }
      fError = 0 ;

      const Char_t *text = newText.Data() ;
      FormSize fs = FirstParse(0,GetTextSize(),text);
      delete[] fTabSize;
      return TMath::Abs(gPad->AbsPixeltoY(Int_t(fs.Height())) - gPad->AbsPixeltoY(0));
}

//______________________________________________________________________________
FormSize TLatex::Readfs()
{
// read fs in fTabSize
      fPos--;
      FormSize result(fTabSize[fPos].width,fTabSize[fPos].dessus,fTabSize[fPos].dessous);
      return result;
}

//______________________________________________________________________________
void TLatex::Savefs(FormSize *fs)
{
// Save fs values in array fTabSize
      fTabSize[fPos].width   = fs->Width();
      fTabSize[fPos].dessus  = fs->Dessus();
      fTabSize[fPos].dessous = fs->Dessous();
      fPos++;
      if (fPos>=fTabMax) {
         // allocate more memory
         FormSize_t *temp = new FormSize_t[fTabMax+100];
         // copy array
         memcpy(temp,fTabSize,fTabMax*sizeof(FormSize_t));
         fTabMax += 100;
         // free previous array
         delete [] fTabSize;
         // swap pointers
         fTabSize = temp;
      }
}

//______________________________________________________________________________
void TLatex::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   if (gROOT->ClassSaved(TLatex::Class())) {
       out<<"   ";
   } else {
       out<<"   TLatex *";
   }
   TString s = GetTitle();
   s.ReplaceAll("\"","\\\"");
   out<<"tex = new TLatex("<<fX<<","<<fY<<","<<quote<<s.Data()<<quote<<");"<<endl;

   SaveTextAttributes(out,"tex",11,0,1,62,1);
   SaveLineAttributes(out,"tex",1,1,1);

   out<<"   tex->Draw();"<<endl;
}

//______________________________________________________________________________
void TLatex::SetIndiceSize(Float_t factorSize)
{
// set relative size of subscripts and superscripts
      fFactorSize = factorSize;
}

//______________________________________________________________________________
void TLatex::SetLimitIndiceSize(Int_t limitFactorSize)
{
// Set limit for text resizing of subscipts and superscripts
      fLimitFactorSize = limitFactorSize;
}
//______________________________________________________________________________
void TLatex::GetTextExtent(UInt_t &w, UInt_t &h, const char *text) const
{
// Return text extent for string text
//  in w return total text width
//  in h return text height
   const UChar_t width12[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     23,32,39,47,47,74,69,20,30,30,47,63,23,31,23,26,47,47,47,47,47,47,47,47,47,47,31,31,63,63,63,47,
     85,57,57,62,67,57,57,67,67,31,41,62,52,77,62,67,57,67,57,47,52,67,57,76,57,52,52,36,26,40,39,47,
     32,47,47,41,47,41,26,47,47,26,26,41,26,67,47,47,47,47,36,36,26,47,41,62,41,41,36,37,26,37,50, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     23,37,47,47,46,47,26,47,28,68,26,47,63,31,69,47,37,51,28,27,31,54,49,31,31,28,29,47,70,70,70,44,
     57,57,57,57,57,57,83,62,57,57,57,57,31,31,31,31,67,62,67,67,67,67,67,63,67,67,67,67,67,52,57,47,
     47,47,47,47,47,47,62,41,41,41,41,41,26,26,26,26,47,47,47,47,47,47,47,51,47,47,47,47,47,41,45,41};
   const UChar_t trail12[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 5, 2, 0, 3, 0, 6,13, 6, 0, 0, 0, 0, 0,15, 1, 0, 0, 0, 2, 1, 4, 4, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 2, 3, 4, 3, 6, 9, 3,14,13,16,11, 1,16,14, 2, 3, 2, 4, 6,12,12,13,13,20,12, 7,11, 0, 7, 4, 4,
      3, 2, 0, 0, 9, 0,36, 7, 0, 2,19, 7, 5, 0, 0, 0,10, 1, 2, 1, 4, 2, 1, 0, 4,10, 4,11, 0, 4, 3, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 4, 2,14, 0, 3, 9, 2,12, 0, 0, 0, 2,12, 3, 1, 4, 4, 5, 0, 0, 3, 0, 4, 9, 0, 0, 0, 0, 0,
      3, 8, 7,13, 9, 8, 7, 4, 6, 6, 6, 6,13,14,14,13, 2,14, 2, 2, 2, 2, 2, 0, 4,12,12,12,12,12, 5,15,
      2, 2, 2, 4, 2, 2, 0, 0, 0, 1, 0, 0, 2, 8, 6, 8, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2,10,11,10};
   const UChar_t height12[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,66,65,67,74,66,66,65,86,86,67,55,22,26,11,68,66,65,65,66,65,65,66,65,66,66,45,56,55,41,55,66,
     89,65,64,66,64,64,64,66,64,64,65,64,64,64,65,66,64,85,64,66,64,65,65,65,64,64,64,84,68,84,65,20,
     67,45,68,45,68,45,87,64,68,63,82,68,68,45,45,45,64,64,44,45,56,45,45,45,45,64,43,87,87,87,32, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,66,78,66,55,64,87,86,61,66,65,45,41,26,66,71,65,54,65,65,65,63,84,38,19,66,65,45,68,68,68,66,
     85,85,84,82,78,77,64,82,85,85,84,78,85,85,84,78,64,83,86,86,85,83,79,52,68,86,86,85,79,85,64,87,
     65,65,64,62,58,64,45,61,65,65,64,58,65,65,64,58,68,62,65,65,64,62,58,50,49,65,65,64,58,84,87,77};

   const UChar_t width22[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     23,31,52,47,47,92,77,26,31,31,47,53,23,31,23,26,47,47,47,47,47,47,47,47,47,47,31,31,53,53,53,47,
     87,67,62,67,67,62,57,72,72,36,47,73,62,88,67,72,57,72,67,52,62,67,67,93,66,67,61,31,26,31,54,47,
     31,47,52,41,52,41,32,47,52,26,31,52,26,77,52,47,52,52,41,36,31,52,47,68,47,47,40,37,20,37,48, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     23,31,47,47,47,47,20,47,34,69,28,47,53,31,69,47,37,51,28,28,31,54,50,31,31,28,31,47,70,70,70,47,
     67,67,67,67,67,67,93,67,62,62,62,62,36,36,36,36,67,67,72,72,72,72,72,53,72,67,67,67,67,67,57,52,
     47,47,47,47,47,47,67,41,41,41,41,41,26,26,26,26,47,52,47,47,47,47,47,51,47,52,52,52,52,47,52,47};
   const UChar_t trail22[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 0, 1, 1, 0, 0, 0, 2, 0, 0, 1, 2, 2, 2, 1, 1, 0, 2, 0, 0, 4,
      0, 0, 0, 0, 0, 0,10, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 1, 2, 1, 1, 0, 0, 0, 0, 3, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 2, 2, 4, 0, 0, 0, 1, 1, 0, 1, 0,
      2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1};
   const UChar_t height22[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,66,65,64,72,66,66,65,84,84,65,56,32,28,17,66,66,65,65,66,65,65,66,64,66,66,46,60,53,43,53,66,
     84,65,64,66,64,64,64,66,64,64,65,64,64,64,65,67,64,81,64,66,64,65,65,65,64,64,64,82,66,82,65,20,
     65,46,65,46,65,46,65,65,64,65,85,64,64,45,45,46,65,65,45,46,61,45,45,45,44,64,44,85,85,85,34, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,66,81,66,55,64,85,85,64,66,65,44,43,28,66,74,65,57,65,65,65,64,84,41,19,65,65,44,68,67,67,66,
     85,85,87,82,83,80,64,82,85,85,87,83,85,85,87,83,64,83,86,86,88,83,84,54,71,86,86,88,84,85,64,65,
     65,65,67,62,63,68,46,62,65,65,67,63,64,64,66,62,66,61,65,65,67,62,63,51,49,65,65,67,63,84,84,82};

   const UChar_t width32[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     23,35,52,46,47,79,72,26,31,31,47,53,23,31,23,26,47,47,47,47,47,47,47,47,47,47,31,31,53,53,53,47,
     77,62,62,62,67,62,62,67,72,36,47,62,57,83,67,67,57,67,62,52,57,67,60,83,62,57,57,31,26,31,53,47,
     31,47,47,41,47,41,31,47,52,26,26,47,26,73,52,47,47,47,36,36,26,52,41,62,47,41,36,32,20,32,54, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     23,36,47,47,46,47,20,47,31,70,25,47,56,31,69,47,37,51,28,28,31,54,47,30,31,28,28,47,69,70,70,47,
     62,62,62,62,62,62,88,62,62,62,62,62,36,36,36,36,67,67,67,67,67,67,67,53,67,67,67,67,67,57,57,47,
     47,47,47,47,47,47,67,41,41,41,41,41,26,26,26,26,47,52,47,47,47,47,47,51,47,52,52,52,52,41,46,41};
   const UChar_t trail32[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 2, 0, 0, 4,12, 7, 2, 3, 0, 1, 0,23, 2, 0, 0, 0, 1, 3, 5, 7, 1, 1, 0, 0, 2, 2, 2, 0,
      6, 3, 3, 7, 4, 7, 9, 4,15,13,11,14, 1,14,15, 2, 8, 2, 2, 7, 9,10,15,12,18,12,11,15, 0,15, 0, 4,
      0, 1, 0, 1, 7, 1,34, 6, 0, 4,17, 3, 6, 0, 0, 0,10, 2, 3, 3, 5, 1, 2, 1, 7,10, 6,15, 0,10, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 5, 2,14, 0, 5, 6, 1,12, 1, 0, 1, 2,16, 4, 2, 6, 5, 8, 4, 6, 5, 0, 3, 9, 0, 0, 0, 0, 0,
      3,10, 9,13,11, 6, 9, 7, 7, 7, 7, 7,13,17,15,14, 1,15, 2, 2, 2, 2, 2, 0, 3,10,10,10,10,12, 3,18,
      1, 2, 1, 3, 1, 1, 2, 1, 1, 5, 4, 5, 0,12, 8, 9, 0, 0, 0, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1,10,11,10};
   const UChar_t height32[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,66,65,65,73,68,66,65,85,85,65,58,31,27,16,66,66,65,65,66,64,65,66,65,66,66,46,60,53,44,53,66,
     85,65,64,66,64,64,64,66,64,64,65,64,64,64,64,66,64,84,64,66,64,65,65,65,64,64,64,82,66,82,65,20,
     65,46,66,46,66,46,85,65,66,66,85,66,66,46,46,46,65,65,45,46,58,46,46,46,46,65,52,84,85,84,34, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,65,80,66,55,64,85,85,62,66,65,44,44,27,66,74,65,58,65,65,65,64,84,38,19,65,65,44,67,67,68,66,
     86,86,86,83,81,80,64,82,86,86,86,81,86,86,86,81,64,83,87,87,87,84,82,52,67,87,87,87,82,86,64,85,
     66,66,66,63,61,69,46,62,66,66,66,61,66,66,66,61,66,63,66,66,66,63,61,51,46,66,66,66,61,85,85,80};

   const UChar_t width42[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     25,26,33,52,52,83,62,18,31,31,36,54,26,31,26,26,52,52,52,52,52,52,52,52,52,52,26,26,54,54,54,52,
     94,62,62,67,67,62,57,72,67,26,47,62,52,77,67,72,62,72,67,62,57,67,62,93,62,61,57,26,26,26,41,52,
     31,52,52,47,52,52,27,52,52,21,21,47,21,79,52,52,52,52,31,47,26,52,47,67,47,45,46,31,23,31,54, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     25,28,52,52,52,52,23,52,31,69,34,52,54,31,69,51,37,51,31,31,31,54,50,31,31,31,34,52,78,78,78,57,
     62,62,62,62,62,62,93,67,62,62,62,62,26,26,26,26,67,67,72,72,72,72,72,54,72,67,67,67,67,61,62,57,
     52,52,52,52,52,52,83,47,52,52,52,52,26,26,26,26,52,52,52,52,52,52,52,51,57,52,52,52,52,45,52,45};
   const UChar_t trail42[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 2, 0, 0, 3, 0, 0, 4,
      0, 0, 0, 1, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 1, 4, 2, 2, 0, 0, 1, 0, 0, 0,
//      0, 0, 0, 1, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 1, 4, 2, 2, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 1, 0,
      3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2};
   const UChar_t height42[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,69,69,69,82,71,71,69,87,87,69,57,24,30,10,69,70,69,69,70,68,69,70,68,70,70,51,65,57,49,57,70,
     90,69,69,71,69,69,69,71,69,69,70,69,69,69,69,71,69,76,69,71,69,70,69,69,69,69,69,87,69,87,70,18,
     69,53,70,53,70,53,70,71,69,69,88,69,69,52,52,53,70,70,52,53,69,52,51,51,51,70,51,89,89,89,42, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,69,88,71,54,69,89,89,69,71,70,47,48,30,71,81,70,58,70,70,68,69,87,40,20,70,70,47,72,71,72,70,
     86,86,86,85,81,83,69,88,86,86,86,81,86,86,86,81,69,85,87,87,87,86,82,57,75,87,87,87,82,86,69,71,
     70,70,70,69,65,74,53,70,70,70,70,65,69,69,69,64,70,68,70,70,70,69,65,53,59,70,70,70,65,88,87,83};

   const UChar_t width52[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     25,26,33,52,52,83,62,18,31,31,36,54,26,31,26,26,52,52,52,52,52,52,52,52,52,52,26,26,54,54,54,52,
     94,62,62,67,67,62,57,72,67,26,47,62,52,77,67,72,62,72,67,62,57,67,62,89,62,62,57,26,26,26,44,52,
     31,52,52,47,52,52,26,52,52,21,21,47,21,76,52,52,52,52,31,47,26,52,47,67,47,47,44,31,24,31,54, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     25,31,52,52,50,52,24,52,31,70,34,52,54,31,70,51,37,51,31,33,31,51,50,31,31,30,34,52,78,78,78,57,
     62,62,62,62,62,62,93,67,62,62,62,62,26,26,26,26,67,67,72,72,72,72,72,54,72,67,67,67,67,62,62,57,
     52,52,52,52,52,52,83,47,52,52,52,52,26,26,26,26,52,52,52,52,52,52,52,51,57,52,52,52,52,47,52,47};
   const UChar_t trail52[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 2,10, 4, 3, 0, 0, 9, 8, 1, 6, 1, 0, 1, 0,20, 3, 0, 2, 2, 1, 4, 3, 6, 3, 1, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 3, 1, 6, 6, 1, 5, 4, 5, 9, 0, 6, 6, 1, 5, 1, 3, 2,11, 5,11,14,16,12, 5,12, 4,14, 5, 5,
      0, 0, 0, 3, 6, 0,14, 4, 0, 7,18, 7, 7, 2, 0, 0, 1, 2,11, 2, 6, 2, 6, 8, 3, 7, 3,13, 1, 7, 1, 0,
//      0, 0, 0, 3, 6, 0,14, 4, 0, 7,18, 7, 7, 2, 0, 0, 1, 2,11, 2, 6, 2, 6, 8, 3, 7, 3,13, 0, 7, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 5, 3,12, 0, 2, 7, 7, 5, 0, 0, 1, 7,12, 2, 0, 8, 7, 7, 6, 8, 4, 0, 6, 6, 0, 4, 2, 5, 0,
      0, 0, 0, 0, 0, 0, 9, 3, 6, 6, 6, 6, 7,12,14,16, 1, 6, 1, 1, 1, 1, 1, 0, 2, 5, 5, 5, 5,12, 3, 0,
      0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 5, 9,10,13, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 7, 0, 7};
   const UChar_t height52[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,69,69,69,79,72,71,69,87,87,69,57,24,30,10,69,70,69,69,70,68,69,70,68,70,70,51,65,57,49,57,70,
     89,69,69,71,69,69,69,71,69,69,70,69,69,69,69,71,69,79,69,71,69,70,69,69,69,69,69,87,69,87,70,18,
     69,53,70,53,70,53,70,71,69,69,88,69,69,52,52,53,70,70,52,53,70,52,51,51,51,70,51,89,87,89,42, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,69,88,71,55,69,87,89,69,71,70,46,49,30,71,81,70,58,70,70,69,69,87,39,22,70,70,46,73,73,73,70,
     85,85,86,84,81,81,69,90,85,85,86,81,85,85,86,81,69,84,86,86,87,85,82,55,81,86,86,87,82,85,69,71,
     69,69,70,68,65,72,54,72,69,69,70,65,68,68,69,64,71,67,69,69,70,68,65,50,59,69,69,70,65,87,87,83};

   const UChar_t width62[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     25,29,44,52,52,81,67,22,31,31,36,54,26,31,26,26,52,52,52,52,52,52,52,52,52,52,31,31,54,54,54,57,
     91,63,67,67,67,62,57,72,67,26,52,67,57,79,67,72,62,72,67,62,57,67,62,87,62,61,57,31,26,31,54,52,
     31,52,57,52,57,52,31,57,57,26,25,52,26,83,57,57,57,57,36,52,31,57,50,74,52,51,47,36,26,36,54, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     25,31,52,52,52,52,26,52,31,69,34,52,54,31,69,51,37,51,31,31,31,54,52,31,30,31,34,52,78,78,78,57,
     63,63,63,63,63,63,93,67,62,62,62,62,26,26,26,26,67,67,72,72,72,72,72,54,72,67,67,67,67,61,62,57,
     52,52,52,52,52,52,83,52,52,52,52,52,26,26,26,26,57,57,57,57,57,57,57,51,57,57,57,57,57,51,57,51};
   const UChar_t trail62[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2, 2, 4, 0, 2, 3, 0, 0, 4,
      0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 1, 1, 2, 1, 0, 0, 2, 0, 0, 0,
//      0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1, 0, 1, 0,
      2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1};
   const UChar_t height62[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,69,69,69,83,73,71,69,87,87,69,57,30,31,15,69,70,69,69,70,68,69,70,68,70,70,51,66,61,51,61,70,
     91,69,69,70,69,69,69,71,69,69,70,69,69,69,69,71,69,76,69,71,69,70,69,69,69,69,69,87,69,87,70,18,
     69,53,70,53,70,53,70,71,69,69,88,69,69,52,52,53,70,70,52,53,69,52,51,51,51,70,51,89,89,89,43, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,69,88,71,59,69,89,90,69,71,70,47,51,31,71,80,70,62,70,70,69,69,87,41,19,70,70,47,70,71,71,70,
     89,89,89,87,86,82,69,87,89,89,89,86,89,89,89,86,69,87,90,90,90,88,87,56,77,90,90,90,87,89,69,71,
     72,72,72,70,69,76,53,69,72,72,72,69,71,71,71,68,70,69,72,72,72,70,69,55,57,72,72,72,69,90,87,87};

   const UChar_t width72[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     25,31,44,52,52,87,67,21,31,30,36,54,26,31,26,26,52,52,52,52,52,52,52,52,52,52,31,31,54,54,54,57,
     91,67,67,67,67,62,57,72,67,26,52,67,57,77,67,72,62,72,67,62,57,67,62,88,62,62,57,31,26,32,54,52,
     31,52,57,52,57,52,31,57,57,26,26,52,26,83,57,57,57,57,36,52,31,57,52,72,52,52,47,36,26,36,54, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     25,31,53,52,52,52,26,52,31,69,34,52,54,31,69,51,37,52,31,31,31,54,51,31,31,31,34,52,78,78,78,57,
     67,67,67,67,67,67,93,67,62,62,62,62,26,26,26,26,67,67,72,72,72,72,72,54,72,67,67,67,67,62,62,57,
     52,52,52,52,52,52,83,52,52,52,52,52,26,26,26,26,57,57,57,57,57,57,57,51,57,57,57,57,57,52,57,52};
   const UChar_t trail72[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 3, 6, 4, 5, 0, 1,11,11, 7, 8, 2, 0, 2, 0,19, 4, 0, 3, 2, 3, 4, 2, 7, 3, 3, 2, 2, 2, 2, 2, 3,
      5, 0, 1, 4, 2, 7, 9, 3, 6, 7, 6, 9, 0, 7, 6, 2, 6, 2, 4, 2,11, 8,14,14,16,14, 7,13, 2,12, 2, 3,
      2, 1, 0, 2, 9, 2,14, 5, 1, 7,20, 7, 7, 2, 0, 0, 1, 4,10, 2, 7, 3, 7,10, 9, 9, 3,11, 0, 7, 2, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 7, 4,13, 0, 3,13, 6, 7, 4, 3, 2, 6,11, 5, 2, 8, 8, 9, 8, 6,11, 0, 5, 6, 1, 4, 2, 4, 0,
      0, 0, 0, 0, 0, 0,11, 4, 7, 7, 7, 7, 8,14,15,21, 2, 6, 2, 2, 2, 2, 2, 0, 3, 8, 8, 8, 8,14, 2, 0,
      1, 1, 1, 1, 4, 1, 2, 2, 2, 2, 2, 4, 5,11,12,17, 2, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 9, 3, 9};
   const UChar_t height72[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,69,69,69,83,71,71,69,87,87,70,58,29,33,15,69,70,69,69,70,68,69,70,68,70,70,51,65,61,51,61,70,
     90,69,69,71,69,69,69,71,69,69,70,69,69,69,69,71,69,78,69,71,69,70,69,69,69,69,69,87,69,87,70,19,
     70,53,70,53,70,53,70,71,69,69,88,69,69,52,52,53,71,71,52,53,68,52,51,51,51,70,51,89,89,89,44, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,69,87,71,59,69,89,90,70,71,70,46,51,33,71,80,70,64,70,70,70,69,88,42,20,70,70,46,71,72,71,70,
     87,87,87,85,86,82,69,88,87,87,87,86,87,87,87,86,69,85,88,88,88,86,87,58,80,88,88,88,87,87,69,71,
     71,71,71,69,70,73,53,70,71,71,71,70,70,70,70,69,70,68,71,71,71,69,70,60,61,71,71,71,70,89,87,88};

   const UChar_t width82[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56};
   const UChar_t trail82[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0,
      1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   const UChar_t height82[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,60,59,69,73,60,52,59,71,71,59,52,29,31,14,73,61,61,60,61,59,60,61,59,61,61,43,50,53,38,53,58,
     66,55,55,57,55,55,55,57,55,55,56,55,55,55,55,57,55,67,55,57,55,56,55,55,55,55,55,71,73,71,61,25,
     62,44,60,44,60,44,59,61,59,62,80,59,59,43,43,44,61,61,43,44,57,43,42,42,42,60,42,71,71,71,35, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,60,62,56,49,55,70,65,59,57,62,45,30,31,58,67,70,58,59,59,62,60,65,37,18,59,62,45,59,59,59,61,
     74,74,73,70,67,76,55,72,74,74,73,67,74,74,73,67,55,70,75,75,74,71,68,45,63,75,75,74,68,74,55,60,
     63,63,62,59,56,65,44,59,63,63,62,56,62,62,61,55,60,58,63,63,62,59,56,49,47,63,63,62,56,80,77,73};

   const UChar_t width92[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56};
   const UChar_t trail92[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 5, 3, 0, 0, 0, 0, 5, 4, 4, 0,
      0, 3, 1, 7, 3, 4,11, 8, 9, 3,11,13, 1,12,12, 6, 7, 4, 3, 3, 8,12,13,11, 8, 9, 2, 0, 0, 0, 0, 4,
      0, 0, 1, 2, 6, 0, 8, 6, 0, 0, 0, 1, 0, 0, 0, 0, 6, 8, 4, 0, 0, 1, 8,10, 2, 5, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 9, 0, 6, 0, 7, 0, 2, 2, 0, 7,16, 0, 2, 0, 0, 0, 1, 7, 0, 0, 0, 0, 4, 5, 8, 5, 0,
      1, 1, 1, 1, 1, 3,11, 7, 4, 4, 4, 4, 3, 3, 3, 3, 1,12, 6, 6, 6, 6, 6, 0,11,12,12,12,12, 9, 3, 0,
      0, 0, 0, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 5, 1, 1, 1, 1, 5, 3, 5};
   const UChar_t height92[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,61,59,69,73,61,52,59,71,71,59,51,29,30,14,73,61,61,60,61,59,60,61,59,61,61,43,52,52,37,52,57,
     67,55,55,57,55,55,55,57,55,55,56,55,55,55,55,57,55,67,55,57,55,56,55,55,55,55,55,71,73,71,60,26,
     62,44,60,44,60,44,59,61,59,62,80,59,59,43,43,44,61,61,43,44,57,43,42,42,42,60,42,71,71,71,35, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,60,63,56,48,55,70,66,59,57,59,42,30,30,58,69,68,55,59,59,62,60,65,37,19,59,59,42,59,60,59,61,
     75,75,74,70,68,80,55,71,75,75,74,68,75,75,74,68,55,70,76,76,75,71,69,44,61,76,76,75,69,75,55,60,
     64,64,63,59,57,69,44,58,64,64,63,57,63,63,62,56,60,58,64,64,63,59,57,50,48,64,64,63,57,81,77,74};

   const UChar_t width102[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56};
   const UChar_t trail102[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
      0, 7, 0, 0, 0, 0, 1, 3, 0, 0, 3, 3, 0, 7, 2, 0, 0, 0, 4, 0, 0, 1, 7, 7, 2, 0, 0, 0, 0, 0, 0,10,
      0, 0, 1, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 1, 3, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7, 1, 0,
      7, 7, 7, 7, 7, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1};
   const UChar_t height102[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,60,55,71,77,60,55,55,72,72,59,52,29,33,14,79,61,60,60,61,59,60,61,59,61,61,42,51,55,41,55,57,
     66,55,55,57,55,55,55,57,55,55,56,55,55,55,55,57,55,70,55,57,55,56,55,55,55,55,55,71,78,71,64,28,
     64,44,60,44,60,44,59,61,59,59,77,59,59,43,43,44,61,61,43,44,58,43,42,42,42,60,42,71,71,71,38, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,60,70,56,49,55,71,67,60,56,62,39,33,33,56,73,70,61,62,62,64,60,67,42,21,62,62,39,62,62,61,60,
     78,78,78,76,72,80,55,72,78,78,78,72,78,78,78,72,55,76,79,79,79,77,73,47,66,79,79,79,73,78,55,60,
     65,65,65,63,59,67,44,59,65,65,65,59,64,64,64,58,60,62,65,65,65,63,59,53,53,65,65,65,59,82,77,76};

  const UChar_t width112[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     55,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,
     56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56};
   const UChar_t trail112[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 4, 4, 3, 2, 0, 3, 0, 2, 5, 0, 0, 0, 7, 2, 0, 3, 3, 0, 4, 8, 6, 4, 4, 0, 0, 8, 8,10, 2,
      4, 7, 5, 9, 8, 7,13, 8,10, 3,12,11, 4,16,11, 7, 4, 7, 6, 5,10,11,12,14, 9, 9, 5, 2, 0, 0, 2,10,
      0, 1, 5, 4, 9, 3,11, 9, 3, 0, 0, 4, 0, 9, 3, 3, 9,11, 7, 1, 3, 3, 9,11, 6,11, 3, 0, 0, 0, 3, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 0, 4,15, 0, 6, 0, 7, 0, 6, 6, 0, 8,19, 0, 5, 0, 0, 1, 3, 9, 0, 0, 0, 0, 4, 7, 9, 8, 0,
      7, 7, 7, 7, 7, 7,16, 9, 7, 7, 7, 7, 3, 4, 3, 3, 8,11, 7, 7, 7, 7, 7, 0,14,11,11,11,11, 9, 5, 0,
      1, 1, 1, 2, 1, 1, 8, 4, 3, 3, 3, 3, 0, 0, 0, 0, 6, 3, 3, 3, 3, 3, 3, 4, 8, 3, 3, 3, 3,11, 9,11};
   const UChar_t height112[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,63,56,75,75,60,55,56,71,71,59,53,29,33,14,79,61,61,60,61,59,60,61,59,61,61,43,52,54,42,54,60,
     67,55,55,57,55,55,55,57,55,55,56,55,55,55,55,57,55,69,55,57,55,56,55,55,55,55,55,71,78,71,64,28,
     64,44,60,44,60,44,59,61,59,59,77,59,59,43,43,44,61,61,43,44,58,43,42,42,42,60,42,71,74,71,38, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,61,68,56,49,55,74,67,59,58,62,41,32,33,58,72,70,65,60,60,60,60,67,39,21,60,62,42,62,64,62,60,
     78,78,77,75,72,81,55,72,78,78,77,72,78,78,77,72,55,75,79,79,78,76,73,48,66,79,79,78,73,78,55,60,
     65,65,64,62,59,68,44,59,65,65,64,59,64,64,63,58,60,61,65,65,64,62,59,53,52,65,65,64,59,82,77,76};

   const UChar_t width122[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0,23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     23,30,66,47,51,78,73,43,31,31,47,51,24,51,23,26,47,47,47,47,47,47,47,47,47,47,25,24,51,51,51,42,
     52,67,63,67,57,57,74,56,68,30,57,68,65,83,67,67,72,68,52,56,58,65,41,73,62,75,57,32,80,32,62,47,
     45,58,54,51,45,41,47,38,56,30,56,49,52,52,49,51,50,48,50,55,40,54,67,61,45,63,48,46,18,46,50, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,57,23,51,15,66,46,71,71,70,71,97,92,56,92,56,37,51,38,51,51,66,45,42,51,52,51,50,94,56,93,61,
     76,64,74,93,71,71,77,71,71,66,65,66,66,67,65,65,71,66,73,73,86,76,51,23,66,56,56,97,92,56,92,56,
     46,34,73,73,70,66,36,36,36,36,36,36,46,46,46,47, 0,34,24,64,64,64,36,36,36,36,36,36,46,46,46, 0};
   const UChar_t trail122[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 2, 0, 2, 1, 0, 0, 0, 1, 1, 0, 4, 1, 1, 2, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2,
     46, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
//     46, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0,34, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,12, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 5, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0};
   const UChar_t height122[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
      2,66,65,65,64,66,66,46,86,86,52,50,27,28,12,68,66,65,65,66,65,65,66,65,66,66,46,61,49,37,49,66,
     47,65,64,64,65,64,64,64,64,64,68,64,65,64,65,66,64,66,64,64,64,64,65,65,64,65,64,84,48,84,65,20,
     87,46,86,65,68,46,87,65,64,46,65,45,68,64,46,46,45,68,64,45,45,46,45,46,87,65,87,86,87,86,32, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,65,65,56,66,45,86,56,56,53,55,50,50,87,50,87,65,59,65,57,45,45,68,45,45,50,40,42,12,107,26,61,
     66,66,67,71,66,66,70,48,48,44,57,58,44,57,44,59,65,66,66,66,65,80,91,31,28,44,44,51,51,87,51,87,
     67,84,66,66,65,80,96,96,96,96,96,96,96,96,96,96, 0,84,96,96,101,96,96,96,96,96,96,96,96,96,96, 0};

   const UChar_t width132[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     23,30,38,47,47,77,72,17,31,31,47,52,23,31,23,26,47,47,47,47,47,47,47,47,47,47,26,26,52,52,52,41,
     85,67,62,62,67,57,52,67,67,31,37,67,56,83,67,67,52,67,62,52,57,67,67,88,67,67,56,32,25,32,44,47,
     30,41,46,41,47,41,31,46,47,26,27,46,26,72,47,46,46,46,31,36,26,47,46,66,47,45,41,45,17,45,50, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     23,30,46,47,47,47,18,47,33,71,25,44,52,31,70,47,36,51,28,27,30,54,42,31,28,28,29,44,69,70,70,42,
     67,67,67,67,67,67,82,62,57,57,57,57,31,31,31,31,67,67,67,67,67,67,67,52,67,67,67,67,67,67,52,46,
     41,41,41,41,41,41,62,41,41,41,41,41,26,26,26,26,47,47,46,46,46,46,46,51,47,47,47,47,47,45,46,45};
   const UChar_t trail132[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 1, 3, 0, 0, 0, 4, 0, 0, 1, 1, 2, 1, 2, 0, 0, 3, 0, 2, 4,
      0, 2, 0, 0, 0, 0,11, 1, 1, 0, 1, 2, 0, 1, 1, 0, 0, 1, 2, 0, 2, 1, 2, 3, 1, 4, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 0,
      2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 4, 0, 4};
   const UChar_t height132[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,66,65,67,76,66,66,65,85,85,67,57,26,27,12,68,66,65,65,66,65,65,66,65,66,66,46,60,55,44,55,66,
     89,65,64,66,64,64,64,66,64,64,65,64,64,64,65,66,64,83,64,66,64,65,65,65,64,64,64,83,68,83,65,20,
     64,46,68,46,68,46,67,65,67,67,87,67,67,45,45,46,65,65,45,46,59,45,45,45,44,64,44,85,87,85,33, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2,66,78,66,54,64,87,83,63,66,65,45,41,27,66,72,65,57,65,65,64,64,84,38,20,65,65,45,66,67,66,66,
     85,85,85,82,80,80,64,83,85,85,85,80,85,85,85,80,64,83,86,86,86,83,81,52,66,86,86,86,81,85,64,67,
     65,65,65,62,60,68,46,63,65,65,65,60,64,64,64,59,68,61,65,65,65,62,60,51,50,65,65,65,60,84,87,79};


   Double_t     wh = (Double_t)gPad->XtoAbsPixel(gPad->GetX2());
   Double_t     hh = (Double_t)gPad->YtoAbsPixel(gPad->GetY1());
   Double_t tsize;
   if (wh < hh)  tsize = fTextSize*wh;
   else          tsize = fTextSize*hh;

   const UChar_t *width ,*trail, *height;

   switch (fTextFont) {
      case 12 :
        width = width12 ; trail = trail12 ; height = height12 ;
        break ;
      case 22 :
        width = width22 ; trail = trail22 ; height = height22 ;
        break ;
      case 32 :
        width = width32 ; trail = trail32 ; height = height32 ;
        break ;
      case 42 :
        width = width42 ; trail = trail42 ; height = height42 ;
        break ;
      case 52 :
        width = width52 ; trail = trail52 ; height = height52 ;
        break ;
      case 62 :
        width = width62 ; trail = trail62 ; height = height62 ;
        break ;
      case 72 :
        width = width72 ; trail = trail72 ; height = height72 ;
        break ;
      case 82 :
        width = width82 ; trail = trail82 ; height = height82 ;
        break ;
      case 92 :
        width = width92 ; trail = trail92 ; height = height92 ;
        break ;
      case 102 :
        width = width102 ; trail = trail102 ; height = height102 ;
        break ;
      case 112 :
        width = width112 ; trail = trail112 ; height = height112 ;
        break ;
      case 122 :
        width = width122 ; trail = trail122 ; height = height122 ;
        break ;
      case 132 :
        width = width132 ; trail = trail132 ; height = height132 ;
        break ;
      default:
        width = width62 ; trail = trail62 ; height = height62 ;
        break ;
   }

   w = h = 0;
   UChar_t ut = 0;
   const char *t = text;
   while(*t) {
      ut = (UChar_t)*t;
      w += width[ut];
      if (height[ut] > h) h = height[ut];
      t++;
   }
   if (ut) w += TMath::Max((UInt_t)trail[ut],(UInt_t)5);
   w = (UInt_t)((w-0)*tsize/100);
   h = (UInt_t)((h+10)*tsize/100);
}

