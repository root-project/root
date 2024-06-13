// Author: Stefan Schmitt
// DESY, 10/08/11

//  Version 17.9, parallel to changes in TUnfold
//
//  History:
//    Version 17.8, relaxed DTD definition
//    Version 17.7, in parallel to changes in TUnfold
//    Version 17.6, with updated doxygen comments
//    Version 17.5, in parallel to changes in TUnfold
//    Version 17.4, in parallel to changes in TUnfoldBinning
//    Version 17.3, support for the "repeat" attribute for element Bin
//    Version 17.2, initial version, numbered in parallel to TUnfold

/** \class TUnfoldBinningXML
XML interfate to binning schemes, for use with the unfolding algorithm
TUnfoldDensity.

Binning schemes are used to map analysis bins on a single histogram
axis and back. The analysis bins may include unconnected bins (e.g
nuisances for background normalisation) or various multidimensional
histograms (signal bins, differential background normalisation bins, etc).
<br/>
If you use this software, please consider the following citation
<br/>
<b>S.Schmitt, JINST 7 (2012) T10003 [arXiv:1205.6201]</b>
<br/>
Detailed documentation and updates are available on
http://www.desy.de/~sschmitt

Please consult the documentation of the class TUnfoldBinning about how to use
binning schemes. This class provides methods to read and write binning
schemes in the XML language. There is also a method which writes out
a dtd file for validation.
<h3>Example XML code</h3>
The example below encodes two binning schemes, <em>detector</em> and
<em>generator</em>. The detecor scheme consists of a single,
three-dimensional distribution (pt,eta,discriminator). The generator
scheme consists of two two-dimensional distributions, signal and background.
<pre>
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE TUnfoldBinning SYSTEM "tunfoldbinning.dtd">
<TUnfoldBinning>
<BinningNode name="detector" firstbin="1" factor="1">
 <BinningNode name="detectordistribution" firstbin="1" factor="1">
  <Axis name="pt" lowEdge="3.5">
   <Bin repeat="3" width="0.5" />
   <Bin repeat="3" width="1" />
   <Bin width="2" />
   <Bin width="3" />
   <Bin location="overflow"/>
   <Axis name="eta" lowEdge="-3">
    <Bin repeat="2" width="0.5" />
    <Bin width="1" />
    <Bin repeat="4" width="0.5" />
    <Bin width="1" />
    <Bin repeat="2" width="0.5" />
    <Axis name="discriminator" lowEdge="0">
     <Bin width="0.15" />
     <Bin repeat="2" width="0.35" />
     <Bin width="0.15" />
    </Axis>
   </Axis>
  </Axis>
 </BinningNode>
</BinningNode>
<BinningNode name="generator" firstbin="1" factor="1">
 <BinningNode name="signal" firstbin="1" factor="1">
  <Axis name="ptgen" lowEdge="4">
   <Bin location="underflow" />
   <Bin width="1" />
   <Bin width="2" />
   <Bin width="3" />
   <Bin location="overflow" />
   <Axis name="etagen" lowEdge="-2">
    <Bin location="underflow" />
    <Bin width="1.5" />
    <Bin width="1" />
    <Bin width="1.5" />
    <Bin location="overflow" />
   </Axis>
  </Axis>
 </BinningNode>
 <BinningNode name="background" firstbin="26" factor="1">
  <Axis name="ptrec" lowEdge="3.5">
   <Bin repeat="3" width="0.5" />
   <Bin repeat="3" width="1" />
   <Bin width="2" />
   <Bin width="3" />
   <Bin location="overflow" />
   <Axis name="etarec" lowEdge="-3">
    <Bin repeat="2" width="0.5" />
    <Bin width="1" />
    <Bin repeat="4" width="0.5" />
    <Bin width="1" />
    <Bin repeat="2" width="0.5" />
   </Axis>
  </Axis>
 </BinningNode>
</BinningNode>
</TUnfoldBinning>
</pre>

*/

/*
  This file is part of TUnfold.

  TUnfold is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  TUnfold is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with TUnfold.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "TUnfold.h"
#include "TUnfoldBinningXML.h"

#include <TXMLDocument.h>
#include <TXMLNode.h>
#include <TXMLAttr.h>
#include <TList.h>
#include <TVectorD.h>

#include <fstream>
#include <sstream>

// #define DEBUG

using std::ofstream, std::stringstream;

ClassImp(TUnfoldBinningXML)

/********************* XML **********************/

////////////////////////////////////////////////////////////////////////
/// write dtd file
///
/// \param[out] out stream for writing the dtd
void TUnfoldBinningXML::WriteDTD(std::ostream &out) {
   out
      <<"<!-- TUnfold Version "<<TUnfold::GetTUnfoldVersion()<<" -->\n"
      <<"<!ELEMENT TUnfoldBinning (BinningNode)+ >\n"
      <<"<!ELEMENT BinningNode ((Bins?,BinningNode*)|(Binfactorlist?,Axis,BinningNode*)) >\n"
      <<"<!ATTLIST BinningNode name ID #REQUIRED firstbin CDATA \"-1\"\n"
      <<"    factor CDATA \"1.\">\n"
      <<"<!ELEMENT Axis ((Bin+,Axis?)|(Axis)) >\n"
      <<"<!ATTLIST Axis name CDATA #REQUIRED lowEdge CDATA #REQUIRED>\n"
      <<"<!ELEMENT Binfactorlist (#PCDATA)>\n"
      <<"<!ATTLIST Binfactorlist length CDATA #REQUIRED>\n"
      <<"<!ELEMENT Bin EMPTY>\n"
      <<"<!ATTLIST Bin width CDATA #REQUIRED location CDATA #IMPLIED\n"
      <<"    center CDATA #IMPLIED repeat CDATA #IMPLIED>\n"
      <<"<!ELEMENT Bins (BinLabel)* >\n"
      <<"<!ATTLIST Bins nbin CDATA #REQUIRED>\n"
      <<"<!ELEMENT BinLabel EMPTY>\n"
      <<"<!ATTLIST BinLabel index CDATA #REQUIRED name CDATA #REQUIRED>\n";
}

////////////////////////////////////////////////////////////////////////
/// write dtd file
///
/// \param[in] file regular file for writing the dtd
void TUnfoldBinningXML::WriteDTD(const char *file) {
   ofstream out(file);
   WriteDTD(out);
}

////////////////////////////////////////////////////////////////////////
/// import a binning scheme from an XML file
///
/// \param[in] document XMP document tree
/// \param[in] name identifier of the binning scheme
///
/// returns a new TUnfoldBinningXML, if <b>name</b> is found in <b>document</b>
TUnfoldBinningXML *TUnfoldBinningXML::ImportXML
(const TXMLDocument *document,const char *name) {
  // import binning scheme from a XML document
  //   document: the XML document
  //   name: the name of the binning scheme to import
  //     if name==nullptr, the first binning scheme found in the tree is imported
   TUnfoldBinningXML *r=nullptr;
   TXMLNode *root=document->GetRootNode();
   TXMLNode *binningNode=nullptr;
   if(root && (!TString(root->GetNodeName()).CompareTo("TUnfoldBinning")) &&
      (root->GetNodeType()==TXMLNode::kXMLElementNode)) {
      // loop over all "BinningNode" entities
      for(TXMLNode *node=root->GetChildren();node && !binningNode;
          node=node->GetNextNode()) {
         if(node->GetNodeType()==TXMLNode::kXMLElementNode &&
            !TString(node->GetNodeName()).CompareTo("BinningNode") &&
            node->GetAttributes()) {
            // localize the BinningNode with the given name
            TIterator *i=node->GetAttributes()->MakeIterator();
            TXMLAttr *attr;
            while((attr=(TXMLAttr *)i->Next())) {
               if((!TString(attr->GetName()).CompareTo("name")) &&
                  ((!TString(attr->GetValue()).CompareTo(name)) ||
                   !name)) {
                  binningNode=node;
               }
            }
         }
      }
   }

   if(binningNode) {
      r=ImportXMLNode(binningNode);
   }
   return r;
}

////////////////////////////////////////////////////////////////////////
/// recursively import one node from the XML tree
///
/// \param[in] node node in the XML document tree
///
/// returns a new TUnfoldBinningXML

TUnfoldBinningXML *TUnfoldBinningXML::ImportXMLNode
(TXMLNode *node) {
   // import data from a given "BinningNode"
   const char *name=nullptr;
   Double_t factor=1.0;
   TUnfoldBinningXML *r=nullptr;
   Int_t nBins=0;
   const char *binNames=nullptr;
   TIterator *i=node->GetAttributes()->MakeIterator();
   TXMLAttr *attr;
   // extract name and global factor
   while((attr=(TXMLAttr *)i->Next())) {
      TString attName(attr->GetName());
      if(!attName.CompareTo("name")) {
         name=attr->GetValue();
      }
      if(!attName.CompareTo("factor")) {
         factor=TString(attr->GetValue()).Atof();
      }
   }
   if(name) {
      TString binNameList="";
      // loop over all children of this BinningNode
      for(TXMLNode *child=node->GetChildren();child;
          child=child->GetNextNode()) {
         // unconnected bins: children are of type "Bins"
         if(child->GetNodeType()==TXMLNode::kXMLElementNode &&
            !TString(child->GetNodeName()).CompareTo("Bins")) {
            // this node has unconnected bins, no axes
            // extract number of bins
            if(child->GetAttributes()) {
               i=child->GetAttributes()->MakeIterator();
               while((attr=(TXMLAttr *)i->Next())) {
                  TString attName(attr->GetName());
                  if(!attName.CompareTo("nbin")) {
                     // number of unconnected bins
                     nBins=TString(attr->GetValue()).Atoi();
                  }
               }
            }
            // extract names of unconnected bins
            TObjArray theBinNames;
            for(TXMLNode *binName=child->GetChildren();binName;
                binName=binName->GetNextNode()) {
               if(binName->GetNodeType()==TXMLNode::kXMLElementNode &&
                  !TString(binName->GetNodeName()).CompareTo("BinLabel")) {
                  i=binName->GetAttributes()->MakeIterator();
                  const char *binLabelName=nullptr;
                  Int_t index=0;
                  while((attr=(TXMLAttr *)i->Next())) {
                     TString attName(attr->GetName());
                     if(!attName.CompareTo("index")) {
                        index=TString(attr->GetValue()).Atoi();
                     }
                     if(!attName.CompareTo("name")) {
                        binLabelName=attr->GetValue();
                     }
                  }
                  if((index>=0)&&(binLabelName)) {
                     if(index>=theBinNames.GetEntriesFast()) {
                        theBinNames.AddAtAndExpand
                           (new TObjString(binLabelName),index);
                     }
                  }
               }
            }
            Int_t emptyName=0;
            for(Int_t ii=0;ii<theBinNames.GetEntriesFast()&&(ii<nBins);ii++) {
               if(theBinNames.At(ii)) {
                  for(Int_t k=0;k<emptyName;k++) binNameList+=";";
                  emptyName=0;
                  binNameList+=
                     ((TObjString *)theBinNames.At(ii))->GetString();
               }
               emptyName++;
            }
            if(binNameList.Length()>0) {
               binNames=binNameList;
            }
         }
      }
      r=new TUnfoldBinningXML(name,nBins,binNames);

      // add add axis information
      r->AddAxisXML(node);

      // import per-bin normalisation factors if there are any
      TVectorD *perBinFactors=nullptr;
      for(TXMLNode *child=node->GetChildren();child;
          child=child->GetNextNode()) {
         // unconnected bins: children are of type "Bins"
         if(child->GetNodeType()==TXMLNode::kXMLElementNode &&
            !TString(child->GetNodeName()).CompareTo("Binfactorlist")) {
            int length=0;
            i=child->GetAttributes()->MakeIterator();
            while((attr=(TXMLAttr *)i->Next())) {
               TString attName(attr->GetName());
               if(!attName.CompareTo("length")) {
                  length=TString(attr->GetValue()).Atoi();
               }
            }
            int nread=0;
            if(length==r->GetDistributionNumberOfBins()) {
               perBinFactors=new TVectorD(length);
               const char *text=child->GetText();
               if(text) {
                  stringstream readFactors(text);
                  for(;nread<length;nread++) {
                     readFactors>> (*perBinFactors)(nread);
                     if(readFactors.fail()) break;
                  }
               }
            }
            if(!perBinFactors) {
               child->Error("ImportXMLNode","while reading per-bin factors"
                            " node=%s length=%d (expected %d)",r->GetName(),
                            length,r->GetDistributionNumberOfBins());
            } else if(nread!=length) {
               child->Error("ImportXMLNode","while reading per-bin factors"
                            " TUnfoldBinning=%s expected %d found %d",
                            r->GetName(),length,nread);
               delete perBinFactors;
               perBinFactors=nullptr;
            }
         }
      }

      // set normalisation factors
      r->SetBinFactor(factor,perBinFactors);

      // now: loop over all child binnings and add them
      for(TXMLNode *child=node->GetChildren();child;
          child=child->GetNextNode()) {
         if(child->GetNodeType()==TXMLNode::kXMLElementNode &&
            !TString(child->GetNodeName()).CompareTo("BinningNode") &&
            child->GetAttributes()) {
            TUnfoldBinning *childBinning=ImportXMLNode(child);
            r->AddBinning(childBinning);
         }
      }
   }
   return r;
}

////////////////////////////////////////////////////////////////////////
/// import axis from XML node
///
/// \param[in] node node in the XML document tree

void TUnfoldBinningXML::AddAxisXML(TXMLNode *node) {
   // find axis if there is one
   TXMLNode *axis=nullptr;
   for(TXMLNode *child=node->GetChildren();child;
       child=child->GetNextNode()) {
      if(child->GetNodeType()==TXMLNode::kXMLElementNode) {
         TString nodeName(child->GetNodeName());
         if(!nodeName.CompareTo("Axis")) axis=child;
      }
   }
   if(axis) {
      const char *axisName=nullptr;
      TArrayD binEdges(1);
      TIterator *i=axis->GetAttributes()->MakeIterator();
      TXMLAttr *attr;
      while((attr=(TXMLAttr *)i->Next())) {
         TString attName(attr->GetName());
         if(!attName.CompareTo("name")) {
            axisName=attr->GetValue();
         }
         if(!attName.CompareTo("lowEdge")) {
            binEdges[0]=TString(attr->GetValue()).Atof();
         }
      }
      Bool_t hasMoreAxes=kFALSE;
      Bool_t underflow=kFALSE,overflow=kFALSE;
      for(TXMLNode *child=axis->GetChildren();child;
          child=child->GetNextNode()) {
         if(child->GetNodeType()==TXMLNode::kXMLElementNode) {
            TString nodeName(child->GetNodeName());
            if(!nodeName.CompareTo("Axis")) hasMoreAxes=kTRUE;
            if(!nodeName.CompareTo("Bin")) {
               Bool_t isUnderflow=kFALSE,isOverflow=kFALSE;
               Int_t repeat=1;
               i=child->GetAttributes()->MakeIterator();
               while((attr=(TXMLAttr *)i->Next())) {
                  TString attName(attr->GetName());
                  TString attText(attr->GetValue());
                  if(!attName.CompareTo("location")) {
                     isUnderflow= !attText.CompareTo("underflow");
                     isOverflow= !attText.CompareTo("overflow");
                  }
                  if(!attName.CompareTo("repeat")) {
                     repeat=attText.Atof();
                  }
               }
               if(repeat<1) {
                  node->Warning("AddAxisXML",
                                "attribute repeat=%d changed to repeat=1",
                                repeat);
                  repeat=1;
               }
               if((isUnderflow || isOverflow)&&(repeat!=1)) {
                  node->Error("AddAxisXML",
     "underflow/overflow can not have repeat!=1 attribute");
               }
               if(isUnderflow || isOverflow) {
                  underflow |= isUnderflow;
                  overflow |= isOverflow;
               } else {
                  Int_t iBin0=binEdges.GetSize();
                  Int_t iBin1=iBin0+repeat;
                  Double_t binWidth=0.0;
                  binEdges.Set(iBin1);
                  i=child->GetAttributes()->MakeIterator();
                  while((attr=(TXMLAttr *)i->Next())) {
                     TString attName(attr->GetName());
                     if(!attName.CompareTo("width")) {
                        binWidth=TString(attr->GetValue()).Atof();
                     }
                  }
                  if(binWidth<=0.0) {
                     node->Error("AddAxisXML",
                                 "bin withd can not be smaller than zero");
                  }
                  for(int iBin=iBin0;iBin<iBin1;iBin++) {
                     binEdges[iBin]=binEdges[iBin0-1]+(iBin-iBin0+1)*binWidth;
                  }
               }
            }
         }
      }
      AddAxis(axisName,binEdges.GetSize()-1,binEdges.GetArray(),
              underflow,overflow);
      if(hasMoreAxes) {
         AddAxisXML(axis);
      }
   }
}

////////////////////////////////////////////////////////////////////////
/// export a binning scheme to a stream in XML format
///
/// \param[in] binning the binning scheme to export
/// \param[out] stream to write to
/// \param[in] writeHeader set true when writing the first binning
/// scheme to this stream
/// \param[in] writeFooter  set true when writing the last binning
/// scheme to this stream
/// \param[in] indent indentation of the XML output
///
/// returns true if the writing succeeded
Int_t TUnfoldBinningXML::ExportXML
(const TUnfoldBinning &binning,std::ostream &out,Bool_t writeHeader,
 Bool_t writeFooter,Int_t indent) {
  //
  if(writeHeader) {
     out<<"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
        <<"<!DOCTYPE TUnfoldBinning SYSTEM \"tunfoldbinning.dtd\">\n"
        <<"<TUnfoldBinning>\n";
  }
  TString trailer(' ',indent);
  out<<trailer<<"<BinningNode name=\""<<binning.GetName()<<"\" firstbin=\""
     <<binning.GetStartBin();
  if(binning.IsBinFactorGlobal()) {
     out<<"\" factor=\""<<binning.GetGlobalFactor()<<"\">\n";
  } else {
     out<<"\">\n";
     out<<trailer<<" <Binfactorlist length=\""
        <<binning.GetDistributionNumberOfBins()<<"\">\n";
     for(int i=0;i<binning.GetDistributionNumberOfBins();i++) {
        if(!(i % 10)) out<<trailer<<" ";
        out<<" "<<binning.GetBinFactor(i+binning.GetStartBin());
        if(((i %10)==9)||(i==binning.GetDistributionNumberOfBins()-1))
           out<<"\n";
     }
     out<<trailer<<" </Binfactorlist>\n";
  }
  if(binning.HasUnconnectedBins()) {
    out<<trailer<<" <Bins nbin=\""<<binning.GetDistributionNumberOfBins()
       <<"\">\n";
    for(Int_t i=0;i<binning.GetDistributionNumberOfBins();i++) {
       const TObjString *name=binning.GetUnconnectedBinName(i);
       if(!name) break;
       out<<trailer<<"  <BinLabel index=\""<<i<<"\" name=\""
	  <<name->GetString()<<"\" />\n";
    }
    out<<trailer<<" </Bins>\n";
  } else {
    for(Int_t axis=0;axis<binning.GetDistributionDimension();axis++) {
      TString axisTrailer(' ',indent+1+axis);
      TVectorD const *edges=binning.GetDistributionBinning(axis);
      out<<axisTrailer<<"<Axis name=\""<<binning.GetDistributionAxisLabel(axis)
	  <<"\" lowEdge=\""<<(*edges)[0]<<"\">\n";
      if(binning.HasUnderflow(axis)) {
	out<<axisTrailer<<" <Bin location=\"underflow\" width=\""
	    <<binning.GetDistributionUnderflowBinWidth(axis)<<"\" center=\""
	    <<binning.GetDistributionBinCenter(axis,-1)<<"\" />\n";
      }
      for(Int_t i=0;i<edges->GetNrows()-1;i++) {
        Int_t repeat=1;
        Double_t width=(*edges)[i+1]-(*edges)[i];
        Double_t center=binning.GetDistributionBinCenter(axis,i);
        for(Int_t j=i+1;j<edges->GetNrows()-1;j++) {
           double xEnd=(j-i+1)*width+(*edges)[i];
           double xCent=center+(j-i)*width;
           if((TMath::Abs(xEnd-(*edges)[j+1])<width*1.E-7)&&
              (TMath::Abs(xCent-binning.GetDistributionBinCenter(axis,j))<
               width*1.E-7)) {
              ++repeat;
           } else {
              break;
           }
        }
        if(repeat==1) {
           out<<axisTrailer<<" <Bin width=\""
              <<width<<"\" center=\""<<center<<"\" />\n";
        } else {
           out<<axisTrailer<<" <Bin repeat=\""<<repeat
              <<"\" width=\""<<width<<"\" center=\""<<center<<"\" />\n";
           i += repeat-1;
        }
      }
      if(binning.HasOverflow(axis)) {
	out<<axisTrailer<<" <Bin location=\"overflow\" width=\""
           <<binning.GetDistributionOverflowBinWidth(axis)<<"\" center=\""
           <<binning.GetDistributionBinCenter(axis,edges->GetNrows()-1)<<"\"/>\n";
      }
    }
    for(Int_t axis=binning.GetDistributionDimension()-1;axis>=0;axis--) {
      TString axisTrailer(' ',indent+1+axis);
      out<<axisTrailer<<"</Axis>\n";
    }
  }
  for(TUnfoldBinning const *child=binning.GetChildNode();child;
      child=child->GetNextNode()) {
     ExportXML(*child,out,kFALSE,kFALSE,indent+1);
  }
  out<<trailer<<"</BinningNode>\n";
  if(writeFooter) {
     out<<"</TUnfoldBinning>\n";
  }
  return out.fail() ? 0 : 1;
}

////////////////////////////////////////////////////////////////////////
/// export this binning scheme to a file
///
/// \param[in] fileName name of the file
///
/// returns true if the writing succeeded
Int_t TUnfoldBinningXML::ExportXML(char const *fileName) const {
  // export this binning scheme to a file
  //   fileName: name of the xml file
  ofstream outFile(fileName);
  Int_t r=ExportXML(*this,outFile,kTRUE,kTRUE);
  outFile.close();
  return r;
}

