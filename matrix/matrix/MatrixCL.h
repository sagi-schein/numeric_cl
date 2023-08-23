#pragma once
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include<CL/cl.h>
#include<memory>
#include<vector>
#include "ClBase.h"

class MatrixCL: public ClBase
{
	cl_context _ctx;
	cl_command_queue _queue;
	cl_mem _matrix_data;
	unsigned int _ncols,_nrows;
	
public:
	MatrixCL();
	MatrixCL(cl_context ctx,cl_command_queue queue, unsigned int nrows,unsigned int ncols,float * data=nullptr);
	~MatrixCL(void);
	MatrixCL(const MatrixCL & that);
	MatrixCL(MatrixCL && that);//move ctor
	std::pair<int,int> dim()const;
	const MatrixCL & operator=(const MatrixCL & that);
	const MatrixCL & operator=(MatrixCL && that);
	MatrixCL operator+(const MatrixCL& right)const;//these are immutable functions
	MatrixCL operator-(const MatrixCL& right)const;
	MatrixCL operator*(const MatrixCL& right)const;
	MatrixCL diag()const;
	MatrixCL diag_inv()const;
	std::vector<float> to_vector() const;


};

