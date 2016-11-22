#include "MatrixCL.h"
#include <iostream>
#include <string>

MatrixCL::MatrixCL():_ncols(0),_nrows(0),_matrix_data(nullptr),_ctx(nullptr),_queue(nullptr)
{
	//create corresponding memory on GPU. There is a trick here, should we get into low GPU memory we might want to back GPU
	//state in main mem and copy on demand. this could be made more ellaborate if we do this on demand
	//when memory becomes limited we may clear 
	
}

MatrixCL::MatrixCL(cl_context ctx,cl_command_queue queue,cl_program program,int nrows,int ncols,float * data): 
	_ctx(ctx),_queue(queue),_program(program),_ncols(ncols),_nrows(nrows),_matrix_data(nullptr)
{
	cl_int err;
	_matrix_data = clCreateBuffer(_ctx, data != nullptr ? CL_MEM_COPY_HOST_PTR : NULL, sizeof(float)*_nrows*_ncols,data, &err);
	if(err != CL_SUCCESS){
		throw std::exception(("failed to create matrix in clCreateBuffer : "+std::to_string(err)).c_str());
	}

}


MatrixCL::~MatrixCL(void)
{
	
	if(_matrix_data != nullptr){
		cl_int res = clReleaseMemObject(_matrix_data);//remember that this will remove a refernce 
		if(res != CL_SUCCESS){
			throw std::exception(("failed to release matrix in clReleaseMemObject : "+std::to_string(res)).c_str());
		}
	}
}



MatrixCL::MatrixCL(const MatrixCL & that)
{
	*this = that;
}

MatrixCL::MatrixCL(MatrixCL && that)
{
	_ctx = that._ctx;
	_queue = that._queue;
	_nrows = that._nrows;
	_ncols = that._ncols;
	_matrix_data = that._matrix_data;
	that._matrix_data = nullptr;

}
//this implements move semantics 
const MatrixCL & MatrixCL::operator=(MatrixCL && that)
{
	_ctx = that._ctx;
	_queue = that._queue;
	_nrows = that._nrows;
	_ncols = that._ncols;
	_matrix_data = that._matrix_data;
	that._matrix_data = nullptr;

	return *this;
}
const MatrixCL& MatrixCL::operator=(const MatrixCL & that)
{
	_ctx = that._ctx;
	_queue = that._queue;
	_nrows = that._nrows;
	_ncols = that._ncols;

	cl_int err;
	_matrix_data = clCreateBuffer(_ctx, NULL, sizeof(float)*_nrows*_ncols,nullptr, &err);
	if(err != CL_SUCCESS){
		throw std::exception(("failed to create matrix in clCreateBuffer : "+std::to_string(err)).c_str());
	}
	err = clEnqueueCopyBuffer(_queue,that._matrix_data,_matrix_data,0,0,sizeof(float)*_nrows*_ncols,0,nullptr,nullptr);
	if(err != CL_SUCCESS){
		throw std::exception(("failed to copy matrix in clEnqueueCopyBuffer : "+std::to_string(err)).c_str());
	}
	err = clEnqueueBarrier(_queue); 
	if(err != CL_SUCCESS){
		throw std::exception(("failed to wait for operation in clEnqueueBarrier : "+std::to_string(err)).c_str());
	}
	return *this;
}

std::pair<int,int> MatrixCL::dim()const
{
	return std::make_pair(_nrows,_ncols);
}


 MatrixCL MatrixCL::operator+(const MatrixCL& right)const
{
	MatrixCL res(_ctx,_queue,_program,_nrows,_ncols,nullptr);
	cl_int err;
	cl_kernel add = clCreateKernel(_program,"add", NULL);
	err = clSetKernelArg(add, 0, sizeof(cl_mem), (void *)&_matrix_data);
	err = clSetKernelArg(add, 1, sizeof(cl_mem), (void *)&right._matrix_data);
	err = clSetKernelArg(add, 2, sizeof(cl_mem), (void *)&res._matrix_data);

	size_t global_work_size[1] = {_nrows*_ncols};
	err = clEnqueueNDRangeKernel(_queue, add, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	//should we wait for completion ?
	return res;
}

MatrixCL MatrixCL::operator-(const MatrixCL& right)const
{
	MatrixCL res(_ctx,_queue,_program,_nrows,_ncols,nullptr);
	return MatrixCL();
}
MatrixCL MatrixCL::operator*(const MatrixCL& right)const
{
	if(_ncols != right._nrows){
		throw std::exception(("matrix dim mismatch : _ncols != right._nrows ("+std::to_string(_ncols)+","+std::to_string(right._nrows)+")").c_str());
	}
	MatrixCL res(_ctx,_queue,_program,_nrows,right._ncols,nullptr);
	cl_int err;
	cl_kernel mult = clCreateKernel(_program,"mult", NULL);
	err = clSetKernelArg(mult, 0, sizeof(cl_mem), (void *)&_matrix_data);
	err = clSetKernelArg(mult, 1, sizeof(cl_mem), (void *)&right._matrix_data);
	err = clSetKernelArg(mult, 2, sizeof(cl_mem), (void *)&res._matrix_data);
	err = clSetKernelArg(mult, 3, sizeof(int), (void *)&_nrows);
	err = clSetKernelArg(mult, 4, sizeof(int), (void *)&_ncols);
	err = clSetKernelArg(mult, 5, sizeof(int), (void *)&right._nrows);
	err = clSetKernelArg(mult, 6, sizeof(int), (void *)&right._ncols);

	size_t global_work_size[1] = {_nrows*right._ncols};
	err = clEnqueueNDRangeKernel(_queue, mult, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	//should we wait for completion ?
	return res;
}
MatrixCL MatrixCL::diag()const
{
	MatrixCL res(_ctx,_queue,_program,_nrows,_ncols,nullptr);
	cl_int err;
	cl_kernel diag = clCreateKernel(_program,"diag", NULL);
	err = clSetKernelArg(diag, 0, sizeof(cl_mem), (void *)&_matrix_data);
	err = clSetKernelArg(diag, 1, sizeof(cl_mem), (void *)&res._matrix_data);
	err = clSetKernelArg(diag, 2, sizeof(int), (void *)&_nrows);
	err = clSetKernelArg(diag, 3, sizeof(int), (void *)&_ncols);
	

	size_t global_work_size[1] = {_nrows*_ncols};
	err = clEnqueueNDRangeKernel(_queue, diag, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	//should we wait for completion ?
	return res;
}
MatrixCL MatrixCL::diag_inv()const
{
	MatrixCL res(_ctx,_queue,_program,_nrows,_ncols,nullptr);
	cl_int err;
	cl_kernel diaginv = clCreateKernel(_program,"diaginv", NULL);
	err = clSetKernelArg(diaginv, 0, sizeof(cl_mem), (void *)&_matrix_data);
	err = clSetKernelArg(diaginv, 1, sizeof(cl_mem), (void *)&res._matrix_data);
	err = clSetKernelArg(diaginv, 2, sizeof(int), (void *)&_nrows);
	err = clSetKernelArg(diaginv, 3, sizeof(int), (void *)&_ncols);
	

	size_t global_work_size[1] = {_nrows*_ncols};
	err = clEnqueueNDRangeKernel(_queue, diaginv, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	//should we wait for completion ?
	return res;
}

std::vector<float> MatrixCL::to_vector() const
{
	std::vector<float> res(_nrows*_ncols);
	cl_int err = clEnqueueReadBuffer(_queue, _matrix_data, CL_TRUE, 0,  res.size()*sizeof(float), res.data(), 0, NULL, NULL);
	return res;
}
