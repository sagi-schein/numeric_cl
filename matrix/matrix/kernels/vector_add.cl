__kernel void add(__global float* left, __global float* right, __global float* out)
{
	int idx = get_global_id(0);
	out[idx] = left[idx]+right[idx];
}