#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include<CL/cl.h>

#include <string>
#include <map>

class ClBase{
        std::map<std::string, std::string> _kernels;
        cl_context _ctx;
    public:
        ClBase(cl_context ctx,std::string kernel_dir);
        ClBase();
        //disable copy ctors, allow just simple construction
        ClBase(const ClBase &) = delete;
        ClBase(ClBase &&) = delete;
        ClBase& operator=(const ClBase &) = delete;

        cl_program getProgram(std::string kernel_name)const;
};