#include "../matrix/matrix/MatrixCL.h"
#include <iostream>
#include <vector>

int main(int argc, char* argv[] ){
    std::cout<<"test matrix_cl - start"<<std::endl;
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;
    cl_uint ret_num_devices = 0;
    cl_platform_id platform_id = 0;
    cl_device_id device_id = 0;
    CL_err = clGetPlatformIDs( 1, &platform_id, &numPlatforms );

    if (CL_err == CL_SUCCESS)
        std::cout<<numPlatforms<<" platform(s) found\n";
    else{
        std::cout<< "clGetPlatformIDs received" << CL_err<<std::endl;
        return 1;
    }
        

    CL_err = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
            &device_id, &ret_num_devices);
    if (CL_err == CL_SUCCESS)
        std::cout<<ret_num_devices<<" devices(s) found\n";
    else {
        std::cout<< "clGetDeviceIDs received" << CL_err<<std::endl;
        return 1;
    }

    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &CL_err);
    if (CL_err == CL_SUCCESS)
        std::cout<<"clCreateContext created valid context\n";
    else {
        std::cout<< "clCreateContext received" << CL_err<<std::endl;
        return 1;
    }
    
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &CL_err);
    if (CL_err == CL_SUCCESS)
        std::cout<<"clCreateCommandQueue created valid command queue\n";
    else {
        std::cout<< "clCreateCommandQueue received" << CL_err<<std::endl;
        return 1;
    }
    ///
    
    ///
    CL_err = clReleaseContext(context);
    if (CL_err == CL_SUCCESS)
        std::cout<<"clReleaseContext done release context\n";
    else {
        std::cout<< "clReleaseContext received" << CL_err<<std::endl;
        return 1;
    }

    std::cout<<"test matrix_cl - finish"<<std::endl;
    return 0;
}