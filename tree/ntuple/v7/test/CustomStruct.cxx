#include "CustomStruct.hxx"

int ComplexStruct::gNCallConstructor = 0;
int ComplexStruct::gNCallDestructor = 0;

ComplexStruct::ComplexStruct()
{
    gNCallConstructor++;
}

ComplexStruct::~ComplexStruct()
{
    gNCallDestructor++;
}

int ComplexStruct::GetNCallConstructor()
{
    return gNCallConstructor;
}

int ComplexStruct::GetNCallDestructor()
{
    return gNCallDestructor;
}

void ComplexStruct::SetNCallConstructor(int n)
{
    gNCallConstructor = n;
}

void ComplexStruct::SetNCallDestructor(int n)
{
    gNCallDestructor = n;
}
