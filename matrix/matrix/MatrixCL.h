#pragma once
#include<CL/cl.h>
#include<memory>
#include<vector>
class MatrixCL
{
	cl_context _ctx;
	cl_command_queue _queue;
	cl_program _program;
	cl_mem _matrix_data;
	int _ncols,_nrows;
	
public:
	MatrixCL();
	MatrixCL(cl_context ctx,cl_command_queue queue, cl_program program,int nrows,int ncols,float * data=nullptr);
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

