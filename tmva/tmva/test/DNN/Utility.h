#ifndef TMVA_TEST_DNN_UTILITY
#define TMVA_TEST_DNN_UTILITY

#include <iostream>
#include <sstream>
#include <type_traits>
#include "stdlib.h"
#include "TRandom.h"
#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/Net.h"

namespace TMVA
{
namespace DNN
{

    template <typename Architecture>
        void constructRandomLinearNet(TNet<Architecture> & net)
    {
        int nlayers = rand() % 5 + 1;

        std::vector<EActivationFunction> ActivationFunctions
            = {EActivationFunction::IDENTITY};

        for (int i = 0; i < nlayers; i++)
        {
            int width = rand() % 10 + 1;
            EActivationFunction f =
                ActivationFunctions[rand() % ActivationFunctions.size()];
            net.AddLayer(width, f);
        }
    }

    /*! Set matrix to the identity matrix */
    template <typename MatrixType>
        void identityMatrix(MatrixType &X)
    {
        size_t m, n;
        m = X.GetNrows();
        n = X.GetNcols();


        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                X(i,j) = 0.0;
            }
            if (i < n) {
                X(i,i) = 1.0;
            }
        }
    }

    /*! Generate a random batch as input for a neural net. */
    template <typename MatrixType>
        void randomMatrix(MatrixType &X)
    {
        size_t m,n;
        m = X.GetNrows();
        n = X.GetNcols();

        TRandom rand(clock());

        Double_t sigma = sqrt(10.0);

        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                X(i,j) = rand.Gaus(0.0, sigma);
            }
        }
    }

    /*! Generate a random batch as input for a neural net. */
    template <typename MatrixType>
        void randomBatch(MatrixType &X)
    {
        randomMatrix(X);
    }

    /*! Generate a random batch as input for a neural net. */
    template <typename MatrixType>
        void copyMatrix(MatrixType &X, const MatrixType &Y)
    {
        size_t m,n;
        m = X.GetNrows();
        n = X.GetNcols();

        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                X(i,j) = Y(i,j);
            }
        }
    }

    /*! Apply functional to each element in the matrix. */
    template <typename MatrixType, typename F>
        void applyMatrix(MatrixType &X, F f)
    {
        size_t m,n;
        m = X.GetNrows();
        n = X.GetNcols();

        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                X(i,j) = f(X(i,j));
            }
        }
    }

    /*! Combine elements of two given matrices into a single matrix using
     *  the given function f. */
    template <typename MatrixType, typename F>
        void zipWithMatrix(MatrixType &Z,
                           F f,
                           const MatrixType &X,
                           const MatrixType &Y)
    {
        size_t m,n;
        m = X.GetNrows();
        n = X.GetNcols();

        for (size_t i = 0; i < m; i++) {for (size_t j = 0; j < n; j++)
            {
                Z(i,j) = f(X(i,j), Y(i,j));
            }
        }
    }

    /*! Generate a random batch as input for a neural net. */
    template <typename MatrixType, typename RealType, typename F>
        RealType reduce(F f, RealType start, const MatrixType &X)
    {
        size_t m,n;
        m = X.GetNrows();
        n = X.GetNcols();

        RealType result = start;

        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                result = f(result, X(i,j));
            }
        }
        return result;
    }

    /*! Generate a random batch as input for a neural net. */
    template <typename MatrixType, typename RealType, typename F>
        RealType reduceMean(F f, RealType start, const MatrixType &X)
    {
        size_t m,n;
        m = X.GetNrows();
        n = X.GetNcols();

        RealType result = start;

        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                result = f(result, X(i,j));
            }
        }
        return result / (RealType) (m * n);
    }

    /*! Compute the relative error of x and y normalized by y. Protected against
     *  division by zero. */
    template <typename RealType>
        inline auto relativeError(const RealType &x,
                                  const RealType &y)
        -> RealType
    {
        if (y != 0.0)
            return std::fabs((x - y) / y);
        else
            return std::fabs(x - y);
    }

    /*! Compute the maximum, element-wise relative error of the matrices
     *  X and Y normalized by the element of Y. Protected against division
     *  by zero. */
    template <typename MatrixType>
        auto maximumRelativeError(const MatrixType &X,
                                  const MatrixType &Y)
        -> decltype(X(0,0))
    {

        using RealType = decltype(X(0,0));

        size_t m,n;
        m = X.GetNrows();
        n = X.GetNcols();

        RealType maximumError = 0.0;

        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                RealType error = relativeError(X(i,j), Y(i,j));
                maximumError = std::max(error, maximumError);
            }
        }
        return maximumError;
    }

    /*! Numerically compute the derivative of the functional f using finite
     *  differences. */
    template <typename F, typename RealType>
        inline RealType finiteDifference(F f, RealType dx)
    {
        return f(dx) - f(0.0 - dx);
    }

    /*! Color code error. */
    template <typename RealType>
        std::string print_error(RealType &e)
    {
        std::ostringstream out{};

        out << ("\e[");

        if (e > 1e-5)
            out << "31m";
        else if (e > 1e-9)
            out << "33m";
        else
            out << "32m";

        out << e;
        out << "\e[39m";

        return out.str();
    }
}
}

#endif
