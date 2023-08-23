

__kernel void sub(__global float* left, __global float* right, __global float* out)
{
	int idx = get_global_id(0);
	out[idx] = left[idx]-right[idx];
}
//left*right -> out
__kernel void mult(__global float* left, __global float* right, __global float* out, 
				   const int nrows_left, 
				   const int ncols_left, 
				   const int nrows_right, 
				   const int ncols_right)
{
	//the math is [nrows_left,ncols_left] X [nrows_right,ncols_right] which means 
	// the size of result matrix is nrows_left * ncols_right and the iner loop is done on  ncols_left (== nrows_right)
	int idx = get_global_id(0);//this is the output index
	int rowidx = idx / ncols_right;
	int colidx = idx % ncols_right;
	float sum =0;
	for(int i = 0;i<ncols_left;i++){
		sum += *(left+rowidx*ncols_left+i)*(*(right+i*ncols_right+colidx));
	}
	out[idx] = sum;
}

//left*right -> out
__kernel void diag(__global float* left, __global float* out, const int ncols, const int nrows)
{
	int idx = get_global_id(0);//this is the output index
	int rowidx = idx / ncols;
	int colidx = idx % ncols;
	if(rowidx == colidx){
		out[idx] = left[idx];
	}
	else
		out[idx] = 0.0f;
}

//left*right -> out
__kernel void diaginv(__global float* left, __global float* out, const int ncols, const int nrows)
{
	int idx = get_global_id(0);//this is the output index
	int rowidx = idx / ncols;
	int colidx = idx % ncols;
	if(rowidx == colidx && left[idx] != 0 ){
		out[idx] = 1.0f/left[idx];
	}
	else
		out[idx] = 0.0f;
}