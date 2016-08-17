#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void hadamard(__global double *B,
                       __global const double *A,
                       int m, int n)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < n) && (y < m)) {
       B[x * m + y] *= A[x * m + y];
    }
}
