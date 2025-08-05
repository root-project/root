#!/usr/bin/python
# -*- coding: utf-8 -*-
## \file
## \ingroup tutorial_graphics
## \notebook
## 
## This script displays all possible types of ROOT/Postscript characters.
## Note: 
##      In pyroot, the encoding needs debugging. Special characters
##      are displayed with an Â-strange-character at the start, like:
##      "\241" -> '¡' appers as "\241" -> "Â¡"
##      Adding the correct -*- coding: utf-8 -*- doesn't help.
##      Changing the kind of font with TText.SetFont doesn't help either.
##      Please, use with caution this script in pyroot. In C-version 
##      This problem doesn't occur.
##
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TText = ROOT.TText
TLine = ROOT.TLine

#types
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
Double_t = ROOT.Double_t
Bool_t = ROOT.Bool_t
char = ROOT.char
c_char_p = ctypes.c_char_p

#C-integration
ProcessLine = ROOT.gInterpreter.ProcessLine

#utils
#strcpy = ROOT.strcpy
def strcpy(destination, source):
   destination = source
   return destination
strcmp = ROOT.strcmp
#sprintf = ROOT.sprintf
def sprintf(buffer, string, *args):
   buffer = string % args
   #print( string % args )
   return buffer
def to_c_char_p(ls):
   c_array = (c_char_p * len(ls) )( )
   c_array[:] = [ c_char_p( item.encode('utf-8') ) for item in ls ]
   return c_array

#prototype functions
def table(x1 : Float_t, x2 : Float_t, yrange : Float_t, t : TText, symbol : char, octal : Bool_t):
   pass

# void
def pstable() :
   global symbol1, symbol2, symbol3, symbol4, symbol5
   #char
   symbol1 = [
       "A","B","C","D","E","F","G","H","I","J","K","L","M","N",
       "O","P","Q","R","S","T","U","V","W","X","Y","Z",
       "0","1","2","3","4","5","6","7","8","9",
       ".",",","+","-","*","/","=","(",")","{","}","END"
       ]

   #char
   symbol2 = [
       "a","b","c","d","e","f","g","h","i","j","k","l","m","n",
       "o","p","q","r","s","t","u","v","w","x","y","z",
       ":",";","@","\\","_","|","%",
       "@'","<",">","[","]","\42","@\43","@\136",
       "@\77","@\41","@&","$","@\176"," ","END"
       ]

   #char
   symbol3 = [
       "\241","\242","\243","\244","\245","\246","\247","\250",
       "\251","\252","\253","\254","\255","\256","\257","\260",
       "\261","\262","\263","\264","\265","\266","\267","\270",
       "\271","\272","\273","\274","\275","\276","\277","\300",
       "\301","\302","\303","\304","\305","\306","\307","\310",
       "\311","\312","\313","\314","\315","\316","\317","END"
       ]

   #char
   symbol4 = [
       "\321","\322","\323","\324","\325","\326","\327","\330",
       "\331","\332","\333","\334","\335","\336","\337","\340",
       "\341","\342","\343","\344","\345","\346","\347","\340",
       "\351","\352","\353","\354","\355","\356","\357","\360",
       "\361","\362","\363","\364","\365","\366","\367","\370",
       "\371","\372","\373","\374","\375","\376","\377","END"
       ]

   #char
   symbol5 = [
       "\177","\200","\201","\202","\203","\204","\205","\206",
       "\207","\210","\211","\212","\213","\214","\215","\216",
       "\217","\220","\221","\222","\223","\224","\225","\226",
       "\227","\230","\231","\232","\233","\234","\235","\236",
       "\237","\240","END"
       ]
   #Note:
   #      Symbol3,4,5 are written in octal-representation: 0o000.
   #      By default, Python recognizes "\177" as its chr-representation
   #      using the utf-8 encoding. "\177" is equivalent to 0o177.
   #      However, Python handles two ways to represent octal numbers; 
   #      0o000 is a 0 and '0o000' is oct(0).
   #      A better writing should be:
   #      IP[in]: 0o000
   #      IP[out]: 0
   #      IP[in]: oct(0)
   #      IP[out]: '0o000' 
   #      Such an importance is to notice how to convert from one to another.
   #      int( '0o000' , base = 8 ) ->  0 # int
   #      int( 0o000, base = 8 ) -> error
   #      oct( 10 ) -> '0o12'
   #      oct( 0o010 ) ->  '0o10' 
   #      Now, coming back to our problem.
   #      "\100" is in an octal number representation, but it is strictly a str. 
   #      It is the '@' character using utf-8. 
   #      To get its octal number:
   #      >>> ord( "\100" )  # 64 #int  
   #      >>> ord( "@" )     # 64 #int 
   #      But it is in a decimal representation, that's why they both don't match
   #      '0o100'
   #      So, in order to get its oct-representation:
   #      >>> oct( ord( "\100" ) ) # '0o100'
   #      >>> oct( ord( "@" ) )    # '0o100'
   #      Which curiously is another str-type. Not to be confused with 
   #      the original "@" a.k.a. "\100".
   #      Because, if we try to get its order: ord('0o100'), python raises error:
   #      >>> TypeError: ord() expected a character, but string of length 5 found.
   #      Do not get lost in the convertions as C does: oct -> char -> oct -> char
   #      are indistintively.
   #      That doesn't occur in Python.
   #      Once we get the str-representation of an escaped octal number: "\100" -> "@" 
   #      Use it and don't try to hand it over to its oct-representation.      
   #      

   
   global c_symbol1, c_symbol2, c_symbol3, c_symbol4, c_symbol5
   c_symbol1 = to_c_char_p( symbol1 )
   c_symbol2 = to_c_char_p( symbol2 )
   c_symbol3 = to_c_char_p( symbol3 )
   c_symbol4 = to_c_char_p( symbol4 )
   c_symbol5 = to_c_char_p( symbol5 )
    
   
   xrange = 18
   yrange = 25
   w = 650
   h = w*yrange//xrange
   
   global c1
   c1 = TCanvas("c1","c1",200,10,w,h)
   c1.Range(0,0,xrange,yrange)
   
   ProcessLine("""
    TText t(0,0,"a");
    t.SetTextSize(0.02);
    //t.SetTextFont(62);
    t.SetTextFont(132);
    t.SetTextAlign(22);
   """)
   global t
   t = ROOT.t
   #Not to use:
   #t = TText(0,0,"a") 
   #t.SetTextSize(0.02)
   ##t.SetTextFont(62)
   #t.SetTextFont(132)
   #t.SetTextAlign(22)
   #Note:
   #      Bad encoding. Every special character appears with an Â strange
   #      at the start, could be a pyroot problem, could be python.
   #      Investigating...
   
   
   table(0.5,0.5*xrange-0.5,yrange,t,symbol1,0)
   table(0.5*xrange+0.5,xrange-0.5,yrange,t,symbol2,0)
   
   global tlabel
   tlabel = TText(0,0,"a")
   #tlabel.SetTextFont(72)
   tlabel.SetTextFont(132)
   tlabel.SetTextSize(0.018)
   tlabel.SetTextAlign(22)
   tlabel.DrawText(0.5*xrange,1.3,
                   "Input characters are standard keyboard characters")
   c1.Modified()
   c1.Update()
   c1.Print("pstable1.ps")
   
   global c2
   c2 = TCanvas("c2","c2",220,20,w,h)
   c2.Range(0,0,xrange,yrange)
   
   table(0.5,0.5*xrange-0.5,yrange,t,symbol3,1)
   table(0.5*xrange+0.5,xrange-0.5,yrange,t,symbol4,1)
   tlabel.DrawText(0.5*xrange,1.3,
   "Input characters using backslash and octal numbers")
   c2.Modified()
   c2.Update()
   c2.Print("pstable2.ps")
   
   global c3
   c3 = TCanvas("c3","c3",240,20,w,h)
   c3.Range(0,0,xrange,yrange)
   
   table(0.5,0.5*xrange-0.5,yrange,t,symbol5,1)
   tlabel.DrawText(0.5*xrange,1.3,
      "Input characters using backslash and octal numbers")
   c3.Modified()
   c3.Update()
   c3.Print("pstable3.ps")
   
#void
def table(x1 : Float_t, x2 : Float_t, yrange : Float_t, t : TText,
          symbol : char, octal : Bool_t) :
   i = Int_t()
   n = 0
   #for (i=0; i<1000; i++) {
   for i in range(0, 1000, 1):
      if (not strcmp(symbol[i],"END")) : break
      n += 1
      
   y1 = 2.5
   y2 = yrange - 0.5
   dx = (x2-x1)/5
   dy = (y2 - 1 -y1)/(n+1)
   y = y2 - 1 - 0.7*dy
   xc0 = x1 + 0.5*dx
   xc1 = xc0 + dx
   xc2 = xc1 + dx
   xc3 = xc2 + dx
   xc4 = xc3 + dx
   
   global line 
   line = TLine()
   line.DrawLine(x1,y1,x1,y2)
   line.DrawLine(x1,y1,x2,y1)
   line.DrawLine(x1,y2,x2,y2)
   line.DrawLine(x2,y1,x2,y2)
   line.DrawLine(x1,y2-1,x2,y2-1)
   line.DrawLine(x1+  dx,y1,x1+  dx,y2)
   line.DrawLine(x1+2*dx,y1,x1+2*dx,y2)
   line.DrawLine(x1+3*dx,y1,x1+3*dx,y2)
   line.DrawLine(x1+4*dx,y1,x1+4*dx,y2)
    
   global txt_italic
   txt_italic = TText(0,0,"a")
   txt_italic.SetTextSize(0.015)
   #txt_italic.SetTextFont(72)
   txt_italic.SetTextFont(132)
   txt_italic.SetTextAlign(22)
   txt_italic.DrawText(xc0,y2-0.6,"Input")
   txt_italic.DrawText(xc1,y2-0.6,"Roman")
   txt_italic.DrawText(xc2,y2-0.6,"Greek")
   txt_italic.DrawText(xc3,y2-0.6,"Special")
   txt_italic.DrawText(xc4,y2-0.6,"Zapf")


   text = " "*12 # char
   #for (i=0; i<n; i++) {
   for i in range(0, n, 1): 
      # First column
      if octal:
         value = oct( ord( symbol[i] ) )
         text = sprintf( text, f"@\\ {value[2:]}")
      else:
         text = strcpy(text, symbol[i])
      t.DrawText(xc0,y,text)

      # Second column.
      text = sprintf(text,"%s", symbol[i])
      t.DrawText(xc1,y,text)

      # Third column.
      text = sprintf(text,"`%s",symbol[i])
      t.DrawText(xc2,y,symbol[i])

      # Fourth column.
      text = sprintf(text,"'%s",symbol[i])
      t.DrawText(xc3,y,text)

      # Fifth column.
      text = sprintf(text,"~%s",symbol[i])
      t.DrawText(xc4,y,text)

      y -= dy
      
   


if __name__ == "__main__":
   pstable()
