#ifndef CPYCPPYY_DIMENSIONS_H
#define CPYCPPYY_DIMENSIONS_H

// Standard
#include <algorithm>
#include <initializer_list>


namespace CPyCppyy {

static const dim_t UNKNOWN_SIZE = (dim_t)-1;

class CPYCPPYY_CLASS_EXPORT Dimensions {
    dim_t* fDims;

public:
    Dimensions(dim_t ndim = 0, dim_t* dims = nullptr) : fDims(nullptr) {
        if (ndim && ndim != UNKNOWN_SIZE) {
            fDims = new dim_t[ndim+1];
            fDims[0] = ndim;
            if (dims) std::copy(dims, dims+ndim, fDims+1);
            else std::fill_n(fDims+1, ndim, UNKNOWN_SIZE);
        }
    }
    Dimensions(std::initializer_list<dim_t> l) {
        fDims = new dim_t[l.size()+1];
        fDims[0] = l.size();
        std::copy(l.begin(), l.end(), fDims+1);
    }
    Dimensions(const Dimensions& d) : fDims(nullptr) {
        if (d.fDims) {
            fDims = new dim_t[d.fDims[0]+1];
            std::copy(d.fDims, d.fDims+d.fDims[0]+1, fDims);
        }
    }
    Dimensions(Dimensions&& d) : fDims(d.fDims) {
        d.fDims = nullptr;
    }
    Dimensions& operator=(const Dimensions& d) {
        if (this != &d) {
            if (!d.fDims) {
                delete [] fDims;
                fDims = nullptr;
            } else {
                if (!fDims || (fDims && fDims[0] != d.fDims[0])) {
                    delete [] fDims;
                    fDims = new dim_t[d.fDims[0]+1];
                }
                std::copy(d.fDims, d.fDims+d.fDims[0]+1, fDims);
            }
        }
        return *this;
    }
    ~Dimensions() {
        delete [] fDims;
    }

public:
    operator bool() const { return (bool)fDims; }

    dim_t ndim() const { return fDims ? fDims[0] : UNKNOWN_SIZE; }
    void ndim(dim_t d) {
        if (fDims) {
            if (fDims[0] == d) return;
            delete [] fDims;
        }

        fDims = new dim_t[d+1];
        fDims[0] = d;
        std::fill_n(fDims+1, d, UNKNOWN_SIZE);
    }

    dim_t  operator[](dim_t i) const { return fDims[i+1]; }
    dim_t& operator[](dim_t i)       { return fDims[i+1]; }

    Dimensions sub() const { return fDims ? Dimensions(fDims[0]-1, fDims+2) : Dimensions(); }
};

typedef Dimensions dims_t;
typedef const dims_t& cdims_t;

} // namespace CPyCppyy

#endif // !CPYCPPYY_DIMENSIONS_H
