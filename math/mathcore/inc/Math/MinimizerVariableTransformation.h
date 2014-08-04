// @(#)root/mathmore:$Id$
// Author: L. Moneta 2009

// Header file for class MinimizerVariable

#ifndef ROOT_Math_MinimizerVariableTransformation
#define ROOT_Math_MinimizerVariableTransformation

namespace ROOT {

   namespace Math {

/**
   Base class for MinimizerVariable transformations defining the functions to deal
   with bounded parameters

   @ingroup MultiMin
*/

class MinimizerVariableTransformation {

public:

   virtual ~MinimizerVariableTransformation() {}

   virtual double Int2ext(double value, double lower, double upper) const = 0;
   virtual double Ext2int(double value, double lower, double upper) const = 0;
   virtual double DInt2Ext(double value, double lower, double upper) const = 0;

};


/**
   Sin Transformation class for dealing with double bounded variables

   @ingroup MultiMin
*/
class SinVariableTransformation : public MinimizerVariableTransformation {

public:

   virtual ~SinVariableTransformation() {}

   double Int2ext(double value, double lower, double upper) const;
   double Ext2int(double value, double lower, double upper) const;
   double DInt2Ext(double value, double lower, double upper) const;

private:


};

/**
   Sqrt Transformation class for dealing with lower bounded variables

   @ingroup MultiMin
*/
class SqrtLowVariableTransformation : public  MinimizerVariableTransformation {
public:

   virtual ~SqrtLowVariableTransformation() {}

   double Int2ext(double value, double lower, double upper) const;
   double Ext2int(double value, double lower, double upper) const;
   double DInt2Ext(double value, double lower, double upper) const;

};

/**
   Sqrt Transformation class for dealing with upper bounded variables

   @ingroup MultiMin
*/
class SqrtUpVariableTransformation : public  MinimizerVariableTransformation {
public:

   virtual ~SqrtUpVariableTransformation() {}

   double Int2ext(double value, double lower, double upper) const;
   double Ext2int(double value, double lower, double upper) const;
   double DInt2Ext(double value, double lower, double upper) const;

};


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_MinimizerVariableTransformation */


