// @(#)root/tmva/pymva $Id$
// Author: Sanjiban Sengupta 2021


#include "TMVA/RModelParser_Keras.h"


namespace TMVA::Experimental::SOFIE::PyKeras {



RModel Parse(std::string /*filename*/, int /* batch_size */ ){

   throw std::runtime_error("TMVA::SOFIE C++ Keras parser is deprecated. Use the python3 function "
                         "model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse('model.keras',batch_size=1) " );

   return RModel();
}
}