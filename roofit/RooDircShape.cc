/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 * File: $Id: RooDircShape.cc,v 1.1.2.5 2002/04/18 23:56:22 zhanglei Exp $
 * Authors:
 *   Lei Zhang, University of Colorado, zhanglei@slac.stanford.edu
 * History:
 *   11-Apr-2002 Lei Zhang, initial version
 *
 * Copyright (C) 2002 University of Colorado
 *****************************************************************************/

#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitModels/RooDircShape.hh"


ClassImp(RooDircShape);


RooDircShape::RooDircShape()
{
}


RooDircShape::RooDircShape(const char *name, const char *title,
			   RooAbsReal &_trkTheta,
			   RooAbsReal &_Bias,
			   TString _coreTail,
			   TString _meanSigma,
			   TString _runSet,
			   TString _dataType,
			   TString _hypothesis,
			   TString _shapefile,
			   Bool_t _useMilliRadian) :
  RooAbsReal(name,title),
  trkTheta("trkTheta", "polar angle", this, _trkTheta),
  Bias("Bias", "offset/mean bias", this, _Bias)
{
  coreTail= _coreTail=="Core"?0:1;
  meanSigma=_meanSigma=="Mean"?0:1;
  // repro-Run1 is treated as Run2 from thetaC parameters point of view
  runSet= _runSet=="Run1"?0:1;
  dataType= _dataType=="Data"?0:1;
  hypothesis= _hypothesis=="Kaon"?0:1;
  mrFactor= _useMilliRadian?1.:.001;
  formParams(_shapefile);
}

RooDircShape::RooDircShape(const RooDircShape& other, const char* name) :
  RooAbsReal(other, name),
  trkTheta("trkTheta", this, other.trkTheta),
  Bias("Bias", this, other.Bias),
  coreTail(other.coreTail), meanSigma(other.meanSigma),
  runSet(other.runSet), dataType(other.dataType),
  hypothesis(other.hypothesis), mrFactor(other.mrFactor)
{
  formParams(other);
}

RooDircShape::~RooDircShape() 
{
}

Double_t RooDircShape::evaluate() const
{
  Int_t bin=getCosThetaBin();
  Double_t myShape=
    ShapeParams[runSet][dataType][hypothesis][coreTail][meanSigma][bin];

  // for mean
  if (0==meanSigma) return (myShape+Bias)*mrFactor;
  
  // for sigma
  // data
  if (0==dataType) return (myShape+Bias)*mrFactor*(0==coreTail?1.:3.);
  // mc
  return dircCubic()*mrFactor*(0==coreTail?1.:3.);
}

Int_t RooDircShape::getCosThetaBin() {

  Int_t number(0);
  Double_t cosTheta(cos(trkTheta));
  
  if (0==runSet && 1==dataType) {
    if (cosTheta >= -1.0 && cosTheta < -0.6) {
      number = 0;
    } else if (cosTheta >= -0.6 && cosTheta < -0.3) {
      number = 1;
    } else if (cosTheta >= -0.3 && cosTheta < 0.0) {
      number = 2;
    } else if (cosTheta >= 0.0 && cosTheta < 0.2) {
      number = 3;
    } else if (cosTheta >= 0.2 && cosTheta < 0.4) {
      number = 4;
    } else if (cosTheta >= 0.4 && cosTheta < 0.6) {
      number = 5;
    } else if (cosTheta >= 0.6 && cosTheta < 0.7) {
      number = 6;
    } else if (cosTheta >= 0.7 && cosTheta < 0.8) {
      number = 7;
    } else {
      number = 8;
    }
  }  else {
    if (cosTheta >= -1.0 && cosTheta < -0.8) {
      number = 0;
    } else if (cosTheta >= -0.8 && cosTheta < -0.6) {
      number = 1;
    } else if (cosTheta >= -0.6 && cosTheta < -0.45) {
      number = 2;
    } else if (cosTheta >= -0.45 && cosTheta < -0.3) {
      number = 3;
    } else if (cosTheta >= -0.3 && cosTheta < -0.15) {
      number = 4;
    } else if (cosTheta >= -0.15 && cosTheta < 0.0) {
      number = 5;
    } else if (cosTheta >= 0.0 && cosTheta < 0.1) {
      number = 6;
    } else if (cosTheta >= 0.1 && cosTheta < 0.2) {
      number = 7;
    } else if (cosTheta >= 0.2 && cosTheta < 0.3) {
      number = 8;
    } else if (cosTheta >= 0.3 && cosTheta < 0.4) {
      number = 9;
    } else if (cosTheta >= 0.4 && cosTheta < 0.5) {
      number = 10;
    } else if (cosTheta >= 0.5 && cosTheta < 0.6) {
      number = 11;
    } else if (cosTheta >= 0.6 && cosTheta < 0.65) {
      number = 12;
    } else if (cosTheta >= 0.65 && cosTheta < 0.7) {
      number = 13;
    } else if (cosTheta >= 0.7 && cosTheta < 0.75) {
      number = 14;
    } else if (cosTheta >= 0.75 && cosTheta < 0.8) {
      number = 15;
    } else if (cosTheta >= 0.8 && cosTheta < 0.85) {
      number = 16;
    } else if (cosTheta >= 0.85 && cosTheta < 0.9) {
      number = 17;
    } else {
      number = 18;
    }
  }
  
  return number;
}

Double_t RooDircShape::dircCubic() {

  Double_t cosTheta(cos(trkTheta));
  
  Double_t *par((Double_t *)&ShapeParams[runSet][dataType][hypothesis][coreTail][meanSigma][0]);
  
  Double_t result=par[0]+cosTheta*(par[1]+cosTheta*(par[2]+cosTheta*par[3]))+
    Bias;
  return result;
}

void RooDircShape::formParams(Double_t *srcParamsPtr) {
  Int_t pSize=sizeof(ShapeParams)/sizeof(Double_t);
  Double_t *parmPtr((Double_t *) ShapeParams);
  for (Int_t i=0; i<pSize; i++) *parmPtr++=*srcParamsPtr++;
}

void RooDircShape::formParams(TString shapefile) {
  
  //             run1/2 data/mc hypo core/tail mean/sigma value
  Double_t myParams [2]    [2]   [2]    [2]       [2]       [19] =
    {
      // run1
      {
	// data
	{
	  // kaon
	  {
	    // core
	    {
	      // mean
	      {-2.084, -0.1905, -0.729, -0.6828, -1.135,
	       -0.7445, -0.5567, -0.4668, -0.1004, -0.2807,
	       -0.4576, -0.4762, -0.6801, -0.3267, -0.4763,
	       -0.1325, -0.1666, -0.4059, -1.202},
	      // sigma
	      {5.565, 2.637, 3.088, 3.163, 3.417,
	       3.540, 3.608, 4.175, 4.143, 3.882,
	       3.586, 3.255, 3.164, 3.187, 3.299,
	       2.945, 2.749, 2.503, 4.042}
	    },
	    // tail
	    {
	      // mean
	      {-2.084, -0.1905, -0.729, -0.6828, -1.135,
	       -0.7445, -0.5567, -0.4668, -0.1004, -0.2807,
	       -0.4576, -0.4762, -0.6801, -0.3267, -0.4763,
	       -0.1325, -0.1666, -0.4059, -1.202},
	      // sigma
	      {5.565, 2.637, 3.088, 3.163, 3.417,
	       3.540, 3.608, 4.175, 4.143, 3.882,
	       3.586, 3.255, 3.164, 3.187, 3.299,
	       2.945, 2.749, 2.503, 4.042}
	    }
	  },
	  // pion
	  {
	    // core
	    {
	      // mean
	      {0.3368, -0.3831, -0.4732, -0.7456, -0.7871,
	       -0.911, -0.3633, -0.2533, -0.3423, -0.5192,
	       -0.5274, -0.7742, -0.9335, -0.9038, -0.3992,
	       -0.2284, -0.4292, -0.7048, -0.9813},
	      // sigma
	      {4.761, 2.813, 3.042, 2.995, 3.457,
	       3.251, 3.547, 3.968, 3.870, 3.788,
	       3.558, 3.542, 3.359, 3.338, 3.205,
	       3.052, 2.873, 2.693, 2.869}
	    }, 
	    // tail
	    {
	      // mean
	      {0.3368, -0.3831, -0.4732, -0.7456, -0.7871,
	       -0.911, -0.3633, -0.2533, -0.3423, -0.5192,
	       -0.5274, -0.7742, -0.9335, -0.9038, -0.3992,
	       -0.2284, -0.4292, -0.7048, -0.9813},
	      // sigma
	      {4.761, 2.813, 3.042, 2.995, 3.457,
	       3.251, 3.547, 3.968, 3.870, 3.788,
	       3.558, 3.542, 3.359, 3.338, 3.205,
	       3.052, 2.873, 2.693, 2.869}
	    }
	  }
	},
	// mc
	{
	  // kaon
	  {
	    // core
	    {
	      // mean
	      {-0.851, -0.613, -0.406, -0.555, -0.508,
	       -0.874, -1.295, -0.796, -1.261},
	      // DircCubic
	      {2.393, 0.7115, 0.01713, -1.514}
	    },
	    // tail
	    {
	      // mean
	      {-0.851, -0.613, -0.406, -0.555, -0.508,
	       -0.874, -1.295, -0.796, -1.261},
	      // DircCubic
	      {2.393, 0.7115, 0.01713, -1.514}
	    }
	  },
	  // pion
	  {
	    // core
	    {
	      // mean
	      {-1.432, -0.843, -0.836, -0.739, -0.593,
	       -1.114, -1.434, -1.324, -1.434},
	      // DircCubic
	      {2.4980, 0.6380, 0.2397, -1.591}
	    },
	    // tail
	    {
	      // mean
	      {-1.432, -0.843, -0.836, -0.739, -0.593,
	       -1.114, -1.434, -1.324, -1.434},
	      // DircCubic
	      {2.4980, 0.6380, 0.2397, -1.591}
	    }
	  }
	}
      },
      // run2
      {
	// data
	{
	  // kaon
	  {
	    // core
	    {
	      // mean
	      {-0.3593, -0.4868, -0.3504, -0.01853, 0.01354,
	       0.03599, +0.6086, 0.113, 0.04001, 0.1525,
	       0.1412, 0.05328, -0.002192, 0.1303, +0.3709,
	       0.1218, 0.01569, -0.3205, -0.9702},
	      // sigma
	      {2.812, 2.693, 2.79, 2.855, 3.054,
	       3.14, 3.19, 3.487, 3.721, +3.624,
	       3.205, 3.063, 2.991, 3.057, 2.863,
	       2.697, 2.445, 2.207, 2.247}
	    },
	    // tail
	    {
	      // mean
	      {-0.3593, -0.4868, -0.3504, -0.01853, 0.01354,
	       0.03599, +0.6086, 0.113, 0.04001, 0.1525,
	       0.1412, 0.05328, -0.002192, 0.1303, +0.3709,
	       0.1218, 0.01569, -0.3205, -0.9702},
	      // sigma
	      {2.812, 2.693, 2.79, 2.855, 3.054,
	       3.14, 3.19, 3.487, 3.721, +3.624,
	       3.205, 3.063, 2.991, 3.057, 2.863,
	       2.697, 2.445, 2.207, 2.247}
	    }
	  },
	  // pion
	  {
	    // core
	    {
	      // mean
	      {-0.2942, -0.2689, -0.3152, -0.04162, 0.2176,
	       0.2536, +0.4403, 0.06096, 0.1115,
	       0.002977, 0.1146, -0.1296, -0.1628, 0.01441,
	       +0.09772, 0.1043, 0.09619, -0.5757, -0.8954},
	      // sigma
	      {2.9, 2.772, 2.982, 2.972, 3.035,
	       2.871, 3.038, 3.447, +3.476, 3.568,
	       3.254, 3.136, 3.015, 3.152, 2.858,
	       2.715, 2.473, 2.352, 3.037}
	    }, 
	    // tail
	    {
	      // mean
	      {-0.2942, -0.2689, -0.3152, -0.04162, 0.2176,
	       0.2536, +0.4403, 0.06096, 0.1115,
	       0.002977, 0.1146, -0.1296, -0.1628, 0.01441,
	       +0.09772, 0.1043, 0.09619, -0.5757, -0.8954},
	      // sigma
	      {2.9, 2.772, 2.982, 2.972, 3.035,
	       2.871, 3.038, 3.447, +3.476, 3.568,
	       3.254, 3.136, 3.015, 3.152, 2.858,
	       2.715, 2.473, 2.352, 3.037}
	    }
	  }
	},
	// mc
	{
	  // kaon
	  {
	    // core
	    {
	      // mean
	      {0.436, 0.380, 0.2139, 0.1339, 0.0260,
	       0.1838, 0.148, -0.1648, -0.2805, -0.1439,
	       -0.230, -0.0502, -0.0206, 0.208, 0.327,
	       0.537, 0.405, 0.3432, 0.6778},
	      // DircCubic
	      {2.393, 0.7115, 0.01713, -1.514}
	    },
	    // tail
	    {
	      // mean
	      {0.436, 0.380, 0.2139, 0.1339, 0.0260,
	       0.1838, 0.148, -0.1648, -0.2805, -0.1439,
	       -0.230, -0.0502, -0.0206, 0.208, 0.327,
	       0.537, 0.405, 0.3432, 0.6778},
	      // DircCubic
	      {2.393, 0.7115, 0.01713, -1.514}
	    }
	  },
	  // pion
	  {
	    // core
	    {
	      // mean
	      {0.674, 0.4648, 0.1435, 0.1625, -0.116,
	       0.124, 0.108, -0.1153, -0.2378, -0.2575,
	       -0.1059, -0.2319, -0.192, 0.030, 0.2272,
	       0.3677, 0.3254, 0.218, 0.131},
	      // DircCubic
	      {2.4980, 0.6380, 0.2397, -1.591}
	    },
	    // tail
	    {
	      // mean
	      {0.674, 0.4648, 0.1435, 0.1625, -0.116,
	       0.124, 0.108, -0.1153, -0.2378, -0.2575,
	       -0.1059, -0.2319, -0.192, 0.030, 0.2272,
	       0.3677, 0.3254, 0.218, 0.131},
	      // DircCubic
	      {2.4980, 0.6380, 0.2397, -1.591}
	    }
	  }
	}
      }
    };

  formParams((Double_t *) myParams);
}

void RooDircShape::formParams(const RooDircShape &other) {
  formParams((Double_t *)other.ShapeParams);
}

Bool_t RooDircShape::isValid() const
{
  return isValid(getVal()) ;
}


Bool_t RooDircShape::isValid(Double_t value, Bool_t verbose) const
{
  return kTRUE ;
}
