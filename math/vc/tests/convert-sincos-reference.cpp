#include <cstdio>

template<typename T> struct SincosReference
{
    const T x, s, c;
};

template<typename T> struct Reference
{
    const T x, ref;
};

template<typename T> struct Data
{
    static const SincosReference<T> sincosReference[];
    static const Reference<T> asinReference[];
    static const Reference<T> acosReference[];
    static const Reference<T> atanReference[];
    static const Reference<T> lnReference[];
    static const Reference<T> log2Reference[];
    static const Reference<T> log10Reference[];
};

namespace Function {
    enum Function {
        sincos, atan, asin, acos, ln, log2, log10
    };
}
template<typename T, Function::Function F> static inline const char *filenameOut();
template<> inline const char *filenameOut<float , Function::sincos>() { return "sincos-reference-single.dat"; }
template<> inline const char *filenameOut<double, Function::sincos>() { return "sincos-reference-double.dat"; }
template<> inline const char *filenameOut<float , Function::atan  >() { return "atan-reference-single.dat"; }
template<> inline const char *filenameOut<double, Function::atan  >() { return "atan-reference-double.dat"; }
template<> inline const char *filenameOut<float , Function::asin  >() { return "asin-reference-single.dat"; }
template<> inline const char *filenameOut<double, Function::asin  >() { return "asin-reference-double.dat"; }
template<> inline const char *filenameOut<float , Function::acos  >() { return "acos-reference-single.dat"; }
template<> inline const char *filenameOut<double, Function::acos  >() { return "acos-reference-double.dat"; }
template<> inline const char *filenameOut<float , Function::ln  >() { return "reference-ln-sp.dat"; }
template<> inline const char *filenameOut<double, Function::ln  >() { return "reference-ln-dp.dat"; }
template<> inline const char *filenameOut<float , Function::log2  >() { return "reference-log2-sp.dat"; }
template<> inline const char *filenameOut<double, Function::log2  >() { return "reference-log2-dp.dat"; }
template<> inline const char *filenameOut<float , Function::log10  >() { return "reference-log10-sp.dat"; }
template<> inline const char *filenameOut<double, Function::log10  >() { return "reference-log10-dp.dat"; }

template<> const SincosReference<float> Data<float>::sincosReference[] = {
#include "sincos-reference-single.h"
};
template<> const SincosReference<double> Data<double>::sincosReference[] = {
#include "sincos-reference-double.h"
};
template<> const Reference<float> Data<float>::asinReference[] = {
#include "asin-reference-single.h"
};
template<> const Reference<double> Data<double>::asinReference[] = {
#include "asin-reference-double.h"
};
template<> const Reference<float> Data<float>::acosReference[] = {
#include "acos-reference-single.h"
};
template<> const Reference<double> Data<double>::acosReference[] = {
#include "acos-reference-double.h"
};
template<> const Reference<float> Data<float>::atanReference[] = {
#include "atan-reference-single.h"
};
template<> const Reference<double> Data<double>::atanReference[] = {
#include "atan-reference-double.h"
};
template<> const Reference<float> Data<float>::lnReference[] = {
#include "reference-ln-sp.h"
};
template<> const Reference<double> Data<double>::lnReference[] = {
#include "reference-ln-dp.h"
};
template<> const Reference<float> Data<float>::log2Reference[] = {
#include "reference-log2-sp.h"
};
template<> const Reference<double> Data<double>::log2Reference[] = {
#include "reference-log2-dp.h"
};
template<> const Reference<float> Data<float>::log10Reference[] = {
#include "reference-log10-sp.h"
};
template<> const Reference<double> Data<double>::log10Reference[] = {
#include "reference-log10-dp.h"
};

template<typename T>
static void convert()
{
    FILE *file;
    file = fopen(filenameOut<T, Function::sincos>(), "wb");
    fwrite(&Data<T>::sincosReference[0], sizeof(SincosReference<T>), sizeof(Data<T>::sincosReference) / sizeof(SincosReference<T>), file);
    fclose(file);

    file = fopen(filenameOut<T, Function::atan>(), "wb");
    fwrite(&Data<T>::atanReference[0], sizeof(Reference<T>), sizeof(Data<T>::atanReference) / sizeof(Reference<T>), file);
    fclose(file);

    file = fopen(filenameOut<T, Function::asin>(), "wb");
    fwrite(&Data<T>::asinReference[0], sizeof(Reference<T>), sizeof(Data<T>::asinReference) / sizeof(Reference<T>), file);
    fclose(file);

    file = fopen(filenameOut<T, Function::acos>(), "wb");
    fwrite(&Data<T>::acosReference[0], sizeof(Reference<T>), sizeof(Data<T>::acosReference) / sizeof(Reference<T>), file);
    fclose(file);

    file = fopen(filenameOut<T, Function::ln>(), "wb");
    fwrite(&Data<T>::lnReference[0], sizeof(Reference<T>), sizeof(Data<T>::lnReference) / sizeof(Reference<T>), file);
    fclose(file);

    file = fopen(filenameOut<T, Function::log2>(), "wb");
    fwrite(&Data<T>::log2Reference[0], sizeof(Reference<T>), sizeof(Data<T>::log2Reference) / sizeof(Reference<T>), file);
    fclose(file);

    file = fopen(filenameOut<T, Function::log10>(), "wb");
    fwrite(&Data<T>::log10Reference[0], sizeof(Reference<T>), sizeof(Data<T>::log10Reference) / sizeof(Reference<T>), file);
    fclose(file);
}

int main()
{
    convert<float>();
    convert<double>();
    return 0;
}
