#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <unistd.h>

#include "log.h"
#include "sinoscope.h"

/* INSPIRE DE STACK OVERFLOW */
#define TOKEN_STR(x) case x: return #x;
const char* opencl_errstr(cl_int err)
{
    switch (err)
    {
        TOKEN_STR(CL_SUCCESS                        )
        TOKEN_STR(CL_DEVICE_NOT_FOUND               )
        TOKEN_STR(CL_DEVICE_NOT_AVAILABLE           )
        TOKEN_STR(CL_COMPILER_NOT_AVAILABLE         )
        TOKEN_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
        TOKEN_STR(CL_OUT_OF_RESOURCES               )
        TOKEN_STR(CL_OUT_OF_HOST_MEMORY             )
        TOKEN_STR(CL_PROFILING_INFO_NOT_AVAILABLE   )
        TOKEN_STR(CL_MEM_COPY_OVERLAP               )
        TOKEN_STR(CL_IMAGE_FORMAT_MISMATCH          )
        TOKEN_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
        TOKEN_STR(CL_BUILD_PROGRAM_FAILURE          )
        TOKEN_STR(CL_MAP_FAILURE                    )
        TOKEN_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
        TOKEN_STR(CL_COMPILE_PROGRAM_FAILURE        )
        TOKEN_STR(CL_LINKER_NOT_AVAILABLE           )
        TOKEN_STR(CL_LINK_PROGRAM_FAILURE           )
        TOKEN_STR(CL_DEVICE_PARTITION_FAILED        )
        TOKEN_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
        TOKEN_STR(CL_INVALID_VALUE                  )
        TOKEN_STR(CL_INVALID_DEVICE_TYPE            )
        TOKEN_STR(CL_INVALID_PLATFORM               )
        TOKEN_STR(CL_INVALID_DEVICE                 )
        TOKEN_STR(CL_INVALID_CONTEXT                )
        TOKEN_STR(CL_INVALID_QUEUE_PROPERTIES       )
        TOKEN_STR(CL_INVALID_COMMAND_QUEUE          )
        TOKEN_STR(CL_INVALID_HOST_PTR               )
        TOKEN_STR(CL_INVALID_MEM_OBJECT             )
        TOKEN_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        TOKEN_STR(CL_INVALID_IMAGE_SIZE             )
        TOKEN_STR(CL_INVALID_SAMPLER                )
        TOKEN_STR(CL_INVALID_BINARY                 )
        TOKEN_STR(CL_INVALID_BUILD_OPTIONS          )
        TOKEN_STR(CL_INVALID_PROGRAM                )
        TOKEN_STR(CL_INVALID_PROGRAM_EXECUTABLE     )
        TOKEN_STR(CL_INVALID_KERNEL_NAME            )
        TOKEN_STR(CL_INVALID_KERNEL_DEFINITION      )
        TOKEN_STR(CL_INVALID_KERNEL                 )
        TOKEN_STR(CL_INVALID_ARG_INDEX              )
        TOKEN_STR(CL_INVALID_ARG_VALUE              )
        TOKEN_STR(CL_INVALID_ARG_SIZE               )
        TOKEN_STR(CL_INVALID_KERNEL_ARGS            )
        TOKEN_STR(CL_INVALID_WORK_DIMENSION         )
        TOKEN_STR(CL_INVALID_WORK_GROUP_SIZE        )
        TOKEN_STR(CL_INVALID_WORK_ITEM_SIZE         )
        TOKEN_STR(CL_INVALID_GLOBAL_OFFSET          )
        TOKEN_STR(CL_INVALID_EVENT_WAIT_LIST        )
        TOKEN_STR(CL_INVALID_EVENT                  )
        TOKEN_STR(CL_INVALID_OPERATION              )
        TOKEN_STR(CL_INVALID_GL_OBJECT              )
        TOKEN_STR(CL_INVALID_BUFFER_SIZE            )
        TOKEN_STR(CL_INVALID_MIP_LEVEL              )
        TOKEN_STR(CL_INVALID_GLOBAL_WORK_SIZE       )
        TOKEN_STR(CL_INVALID_PROPERTY               )
        TOKEN_STR(CL_INVALID_IMAGE_DESCRIPTOR       )
        TOKEN_STR(CL_INVALID_COMPILER_OPTIONS       )
        TOKEN_STR(CL_INVALID_LINKER_OPTIONS         )
        TOKEN_STR(CL_INVALID_DEVICE_PARTITION_COUNT )
        default: return "Unknown OpenCL error code";
    }
}

// second paramètre est une structure contenant toutes les valeurs entières de sinoscope_t et
typedef struct sinoscope_int{
	unsigned int buffer_size;
    unsigned int width;
    unsigned int height;
    unsigned int taylor;
	unsigned int interval;
} sinoscope_int_t;

static sinoscope_int_t second_param;

//troisième paramètre est une structure contenant toutes valeurs à virgule flotante
typedef struct sinoscope_float{
	float interval_inverse;
	float time;
    float max;
    float phase0;
    float phase1;
    float dx;
    float dy;
} sinoscope_float_t;


static sinoscope_float_t third_param;

int sinoscope_opencl_init(sinoscope_opencl_t* opencl, cl_device_id opencl_device_id, unsigned int width, unsigned int height) {

	cl_ulong mem_size;
	clGetDeviceInfo(opencl_device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
	LOG_ERROR("GLOBAL MEMORY SIZE: %lu MB", mem_size / (1024 * 1024));

	opencl->device_id=opencl_device_id;
    cl_int errcode_ret = 0;
    opencl->context = clCreateContext(NULL, 1, &opencl->device_id, NULL, NULL, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        LOG_ERROR("Failed to create OpenCL context");
        return -1;
    }

	opencl->queue = clCreateCommandQueue (opencl->context, opencl->device_id, 0, &errcode_ret);
	if (errcode_ret != CL_SUCCESS) {
        LOG_ERROR("Failed to create OpenCL queue");
        return -1;
    }

	size_t size = (width * height) * 3 ;
	opencl->buffer = clCreateBuffer (opencl->context, CL_MEM_WRITE_ONLY, size, NULL, &errcode_ret);
	if (errcode_ret != CL_SUCCESS) {
        LOG_ERROR("Failed to create OpenCL buffer");
        return -1;
    }

	char *shader_source = NULL;
	size_t shader_len = 0;
	if(opencl_load_kernel_code(&shader_source, &shader_len) < 0){
		LOG_ERROR("FAILED TO LOAD KERNEL CODE");
		return -1;
	}

	cl_program program = clCreateProgramWithSource(opencl->context, 1, (const char **)&shader_source, &shader_len, &errcode_ret);
	free(shader_source);
	if (errcode_ret != CL_SUCCESS) {
        LOG_ERROR("Failed to create OpenCL program");
        return -1;
    }

	char options[512];
	sprintf(options, "-I %s", __OPENCL_INCLUDE__);
	cl_int build_status = clBuildProgram (program, 1, &opencl->device_id, options, NULL, NULL);
	if(opencl_print_build_log(program, opencl->device_id) < 0){
		LOG_ERROR("FAILED TO PRINT KERNEL BUILD LOGS");
		return -1;
	}
	if(build_status != CL_SUCCESS){
		LOG_ERROR("FAILED TO BUILD KERNEL PROGRAM: %s", opencl_errstr(build_status));
        return -1;
	}

	const char *kernel_name = "sinoscope_kernel";
	opencl->kernel = clCreateKernel(program, kernel_name, &errcode_ret);
	if (errcode_ret != CL_SUCCESS) {
        LOG_ERROR("Failed to create OpenCL buffer");
        return -1;
    }

	// set kernel args
	// first the buffer? why set it here when wze set it also in exec
	// OpenCL, a buffer is a memory object that stores data that can be read or written by kernels. Buffers are used to transfer data between the host (CPU) and the device (GPU or other OpenCL-capable devices).
	errcode_ret = clSetKernelArg(opencl->kernel, 0, sizeof(cl_mem), &opencl->buffer);
	if (errcode_ret != CL_SUCCESS){
		LOG_ERROR("FAILED TO ADD ARG 0 TO KERNEL: %s", opencl_errstr(errcode_ret));
		return -1;
	}

	// set kernel arg of sinoscope split between int and float
	// why set it here when we set it also in exec
	errcode_ret = clSetKernelArg(opencl->kernel, 1, sizeof(sinoscope_int_t), &second_param);
	if (errcode_ret != CL_SUCCESS){
		LOG_ERROR("FAILED TO ADD ARG 1 TO KERNEL: %s", opencl_errstr(errcode_ret));
		return -1;
	}

	errcode_ret = clSetKernelArg(opencl->kernel, 2, sizeof(sinoscope_float_t), &third_param);
	if (errcode_ret != CL_SUCCESS){
		LOG_ERROR("FAILED TO ADD ARG 2 TO KERNEL: %s", opencl_errstr(errcode_ret));
		return -1;
	}

	return 0;
}

void sinoscope_opencl_cleanup(sinoscope_opencl_t* opencl)
{
	clReleaseKernel(opencl->kernel);
	clReleaseCommandQueue(opencl->queue);
	clReleaseContext(opencl->context);
	clReleaseMemObject(opencl->buffer);
}

int sinoscope_image_opencl(sinoscope_t* sinoscope) {
	second_param.buffer_size = sinoscope->buffer_size;
	second_param.width = sinoscope->width;
	second_param.height = sinoscope->height;
	second_param.taylor = sinoscope->taylor;
	second_param.interval = sinoscope->interval;

	third_param.interval_inverse = sinoscope->interval_inverse;
	third_param.time = sinoscope->time;
	third_param.max = sinoscope->max;
	third_param.phase0 = sinoscope->phase0;
	third_param.phase1 = sinoscope->phase1;
	third_param.dx = sinoscope->dx;
	third_param.dy = sinoscope->dy;

	// set kernel arg of sinoscope type (could be splittedd between int and float?)
	cl_int errcode_ret = clSetKernelArg(sinoscope->opencl->kernel, 1, sizeof(sinoscope_int_t), &second_param);
	if (errcode_ret != CL_SUCCESS){
		LOG_ERROR("FAILED TO ADD ARG 1 TO KERNEL: %s", opencl_errstr(errcode_ret));
		return -1;
	}

	errcode_ret = clSetKernelArg(sinoscope->opencl->kernel, 2, sizeof(sinoscope_float_t), &third_param);
	if (errcode_ret != CL_SUCCESS){
		LOG_ERROR("FAILED TO ADD ARG 2 TO KERNEL: %s", opencl_errstr(errcode_ret));
		return -1;
	}

	// enqueue and set optimal work group size, when wg null => let opencl deal with it
	// Enqueues a command to execute a kernel on a device.
	// total_size = global_work_size => number of work Item I want to compute,  the total number of work-items that can execute this kernel in parallel
	// we want : répartition du calcul doit se faire en deux dimensions
	const size_t total_size[2] = {sinoscope->width, sinoscope->height};
	errcode_ret = clEnqueueNDRangeKernel(sinoscope->opencl->queue, sinoscope->opencl->kernel, 1, NULL, total_size, NULL, 0, NULL, NULL);
	if (errcode_ret != CL_SUCCESS){
		LOG_ERROR("FAILED TO ENQUEUE OPENCL KERNEL: %s", opencl_errstr(errcode_ret));
		return -1;
	}

	// clEnqueueReadBuffer : To read from a buffer object to host memory or to write to a buffer object from host memory call one of the functions
	// sinoscope->buffer pointer to buffer in host memory where data is to be read into or to be written from
	errcode_ret = clEnqueueReadBuffer(sinoscope->opencl->queue, sinoscope->opencl->buffer, CL_TRUE, 0, sinoscope->buffer_size, sinoscope->buffer, 0, NULL, NULL);
	if (errcode_ret != CL_SUCCESS){
		LOG_ERROR("FAILED TO ENQUEUE READ IMAGE: %s", opencl_errstr(errcode_ret));
		return -1;
	}

	return 0;
}
