// @(#)root/tmva/pymva $Id$
// Author: Sanjiban Sengupta 2021


#include "TMVA/RModelParser_Keras.h"


namespace TMVA::Experimental::SOFIE::PyKeras {



RModel Parse(std::string /*filename*/, int /* batch_size */ ){

   throw std::runtime_error("TMVA::SOFIE C++ Keras parser is deprecated. Use python3 function "
                         "ROOT.TMVA.Experimental.SOFIE.RModelParser_Keras.Parse('model.keras',batch_size) " );

   return RModel();
}
}