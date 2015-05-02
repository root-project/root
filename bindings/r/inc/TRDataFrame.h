// @(#)root/r:$Id$
// Author: Omar Zapata   30/05/2015


/*************************************************************************
 * Copyright (C) 2013-2015, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRDataFrame
#define ROOT_R_TRDataFrame

#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

#ifndef ROOT_R_TRObjectProxy
#include<TRObjectProxy.h>
#endif

//________________________________________________________________________________________________________
/**
   This is a base class to create DataFrames from ROOT to R


   @ingroup R
*/

namespace ROOT {
namespace R {

static Rcpp::internal::NamedPlaceHolder Label;

class TRDataFrame: public TObject {
    friend class TRInterface;
    friend SEXP Rcpp::wrap<TRDataFrame>(const TRDataFrame &f);
protected:
    Rcpp::DataFrame df;
public:
    //Proxy class to use operators for assignation Ex: df["name"]>>object
    class Binding {
        friend class TRDataFrame;
    public:
        Binding(Rcpp::DataFrame &_df,TString name):fName(name),fDf(_df) {}
        Binding(const Binding &obj):fName(obj.fName),fDf(obj.fDf){}
        template <class T> Binding operator=(T var) {
            int size = fDf.size(),i=0 ;
            Rcpp::CharacterVector names=fDf.attr("names");
            bool found=false;
            while(i<size)
            {
                if(names[i]==fName.Data())
                {
                    found=true;
                    break;
                }
                i++;
            }
            if(found) fDf[fName.Data()]=var;
            else
            {
                Rcpp::List nDf(size+1);
                Rcpp::CharacterVector nnames(size+1);
                for(i=0; i<size; i++) {
                    nDf[i] = fDf[i] ;
                    nnames[i] = names[i];
                }
                nDf[size]=var;
                nnames[size]=fName.Data();

                nDf.attr("class") = fDf.attr("class") ;
                nDf.attr("row.names") = fDf.attr("row.names") ;
                nDf.attr("names") = nnames ;
                fDf=nDf;
            }
            return *this;
        }
        Binding operator=(Binding obj)
        {
            int size = fDf.size(),i=0 ;
            Rcpp::CharacterVector names=fDf.attr("names");
            bool found=false;
            while(i<size)
            {
                if(names[i]==fName.Data())
                {
                    found=true;
                    break;
                }
                i++;
            }
            if(found) fDf[fName.Data()]=obj.fDf[obj.fName.Data()];
            else
            {
                Rcpp::List nDf(size+1);
                Rcpp::CharacterVector nnames(size+1);
                for(i=0; i<size; i++) {
                    nDf[i] = obj.fDf[i] ;
                    nnames[i] = names[i];
                }
                nDf[size]=obj.fDf[obj.fName.Data()];
                nnames[size]=fName.Data();

                nDf.attr("class") = obj.fDf.attr("class") ;
                nDf.attr("row.names") = obj.fDf.attr("row.names") ;
                nDf.attr("names") = nnames ;
                fDf=nDf;
            }
            
            return *this;
        }
       
        template <class T> Binding &operator >>(T &var) {
            var = Rcpp::as<T>(fDf[fName.Data()]);
            return *this;
        }
        Binding operator >>(Binding var) {
            var.fDf[var.fName.Data()] = fDf[fName.Data()];
            return var;
        }

        template <class T> Binding &operator <<(T var) {
            int size = fDf.size(),i=0 ;
            Rcpp::CharacterVector names=fDf.attr("names");
            bool found=false;
            while(i<size)
            {
                if(names[i]==fName.Data())
                {
                    found=true;
                    break;
                }
                i++;
            }
            if(found) fDf[fName.Data()]=var;
            else
            {
                Rcpp::List nDf(size+1);
                Rcpp::CharacterVector nnames(size+1);
                for(i=0; i<size; i++) {
                    nDf[i] = fDf[i] ;
                    nnames[i] = names[i];
                }
                nDf[size]=var;
                nnames[size]=fName.Data();

                nDf.attr("class") = fDf.attr("class") ;
                nDf.attr("row.names") = fDf.attr("row.names") ;
                nDf.attr("names") = nnames ;
                fDf=nDf;
            }
            return *this;
        }
        template <class T> operator T() {
            return Rcpp::as<T>(fDf[fName.Data()]);
        }
        template <class T> operator T() const{
            return Rcpp::as<T>(fDf[fName.Data()]);
        }

    private:
        TString fName;
        Rcpp::DataFrame &fDf;
    };

    TRDataFrame();
    TRDataFrame(const TRDataFrame &_df);
    TRDataFrame(const Rcpp::DataFrame &_df):df(_df){};
#include <TRDataFrame__ctors.h>
    Binding operator[](const TString &name);
    
    TRDataFrame& operator=(TRDataFrame &obj) {
            df=obj.df;
            return *this;
         }
    TRDataFrame& operator=(TRDataFrame obj) {
            df=obj.df;
            return *this;
         }
    int GetNcols(){return df.size();}
    int GetNrows(){return df.nrows();}
    ClassDef(TRDataFrame, 0) //
};
}
}



#endif
