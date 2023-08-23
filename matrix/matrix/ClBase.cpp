#include <filesystem>
#include <string>
#include <fstream>
#include <streambuf>
#include "ClBase.h"
namespace fs = std::filesystem;
#include <iostream>

ClBase::ClBase():_ctx(nullptr){}
ClBase::ClBase(cl_context ctx,std::string kernel_dir) : _ctx(ctx){

    for (const auto & entry : fs::directory_iterator(kernel_dir)){
        std::cout<<entry.path()<<std::endl;    
        std::ifstream t(entry.path().c_str());
        std::string kernel;

        t.seekg(0, std::ios::end);   
        kernel.reserve(t.tellg());
        t.seekg(0, std::ios::beg);

        kernel.assign((std::istreambuf_iterator<char>(t)),
                std::istreambuf_iterator<char>());    
        _kernels.insert({entry.path().filename().string(),kernel});
    }
    
 }

 cl_program ClBase::getProgram(std::string kernel_name) const{

    cl_int ret;    
    if(_kernels.find(kernel_name) == _kernels.end()){
        return nullptr;
    }
    auto prog_len = _kernels[kernel_name].length();
    auto prog_str = _kernels[kernel_name].c_str();

     // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(_ctx, 1, 
            (const char **)&(prog_str), (const size_t *)&prog_len, &ret);
    if(ret == CL_SUCCESS){
        return program;
    }        
    else{
        return nullptr;
    }

 }
