#include "operators.h"

// for testing the case of virtual operator==
v_opeq_base::v_opeq_base(int val) : m_val(val) {}
v_opeq_base::~v_opeq_base() {}

bool v_opeq_base::operator==(const v_opeq_base& other) {
   return m_val == other.m_val;
}

v_opeq_derived::v_opeq_derived(int val) : v_opeq_base(val) {}
v_opeq_derived::~v_opeq_derived() {}

bool v_opeq_derived::operator==(const v_opeq_derived& other) {
   return m_val != other.m_val;
}


// for indexing tests
int& YAMatrix1::operator() (int, int) {
    return m_val;
}

const int& YAMatrix1::operator() (int, int) const {
    return m_val;
}

//-
int& YAMatrix2::operator[] (int) {
    return m_val;
}

const int& YAMatrix2::operator[] (int) const {
    return m_val;
}

//-
int& YAMatrix3::operator() (int, int) {
    return m_val;
}

const int& YAMatrix3::operator() (int, int) const {
    return m_val;
}

int& YAMatrix3::operator[] (int) {
    return m_val;
}

const int& YAMatrix3::operator[] (int) const {
    return m_val;
}

//-
int& YAMatrix4::operator[] (int) {
    return m_val;
}

const int& YAMatrix4::operator[] (int) const {
    return m_val;
}

int& YAMatrix4::operator() (int, int) {
    return m_val;
}

const int& YAMatrix4::operator() (int, int) const {
    return m_val;
}

//-
int& YAMatrix5::operator[] (int) {
    return m_val;
}

int YAMatrix5::operator[] (int) const {
    return m_val;
}

int& YAMatrix5::operator() (int, int) {
    return m_val;
}

int YAMatrix5::operator() (int, int) const {
    return m_val;
}

//-
int& YAMatrix6::operator[] (int) {
    return m_val;
}

int YAMatrix6::operator[] (int) const {
    return m_val;
}

int& YAMatrix6::operator() (int, int) {
    return m_val;
}

//-
int& YAMatrix7::operator[] (int) {
    return m_val;
}

int& YAMatrix7::operator() (int, int) {
    return m_val;
}

