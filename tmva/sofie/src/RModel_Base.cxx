#include <limits>
#include <algorithm>
#include <cctype>
#include "TMVA/RModel_Base.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

RModel_Base::RModel_Base(std::string name, std::string parsedtime):fFileName(name), fParseTime(parsedtime) {
    fName = fFileName.substr(0, fFileName.rfind("."));
    fName = UTILITY::Clean_name(fName);
}

void RModel_Base::GenerateHeaderInfo(std::string& hgname) {
    fGC += ("//Code generated automatically by TMVA for Inference of Model file [" + fFileName + "] at [" + fParseTime.substr(0, fParseTime.length()-1) +"] \n");
    // add header guards
    hgname = fName;
    std::transform(hgname.begin(), hgname.end(), hgname.begin(), [](unsigned char c) {
                       return std::toupper(c);
                   } );
    hgname = "ROOT_TMVA_SOFIE_" + hgname;
    fGC += "\n#ifndef " + hgname + "\n";
    fGC += "#define " + hgname + "\n\n";
    for (auto& i: fNeededStdLib) {
        fGC += "#include <" + i + ">\n";
    }
    for (auto& i: fCustomOpHeaders) {
        fGC += "#include \"" + i + "\"\n";
    }
    // for the session we need to include SOFIE_Common functions
    //needed for convolution operator (need to add a flag)
    fGC += "#include \"TMVA/SOFIE_common.hxx\"\n";
    if (fUseWeightFile)
        fGC += "#include <fstream>\n";
    // Include TFile when saving the weights in a binary ROOT file
    if (fWeightFile == WeightFileType::RootBinary)
        fGC += "#include \"TFile.h\"\n";

    fGC += "\nnamespace TMVA_SOFIE_" + fName + "{\n";
    if (!fNeededBlasRoutines.empty()) {
        fGC += ("namespace BLAS{\n");
        for (auto &routine : fNeededBlasRoutines) {
            if (routine == "Gemm") {
                fGC += ("\textern \"C\" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,\n"
                        "\t                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,\n"
                        "\t                       const float * beta, float * C, const int * ldc);\n");
            } else if (routine == "Gemv") {
                fGC += ("\textern \"C\" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,\n"
                        "\t                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);\n");
            } else if (routine == "Axpy") {
                fGC += ("\textern \"C\" void saxpy_(const int * n, const float * alpha, const float * x,\n"
                        "\t                         const int * incx, float * y, const int * incy);\n");
            } else if (routine == "Copy") {
                fGC += ("\textern \"C\" void scopy_(const int *n, const float* x, const int *incx, float* y, const int* incy);\n");
            }
        }
        fGC += ("}//BLAS\n");
    }
}

void RModel_Base::OutputGenerated(std::string filename, bool append) {
    // the model can be appended only if a file name is provided
    if (filename.empty()) {
        // if a file is pr
        filename = fName + ".hxx";
        append = false;
    }
    std::ofstream f;
    if (append)
        f.open(filename, std::ios_base::app);
    else
        f.open(filename);
    if (!f.is_open()) {
        throw std::runtime_error("tmva-sofie failed to open file for output generated inference code");
    }
    f << fGC;
    f.close();
}

}//SOFIE
}//Experimental
}//TMVA
