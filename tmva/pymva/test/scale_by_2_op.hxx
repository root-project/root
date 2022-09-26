#include <vector>
#include <algorithm>
#include <iostream>
namespace Scale_by_2{
void Compute(const std::vector<float>& input, std::vector<float>& output){
    for(size_t i=0; i<input.size(); ++i){
        output[i]=input[i]*2;
    }
}
} //Scale_by_2 operator