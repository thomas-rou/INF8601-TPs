#include <stdio.h>

#define SERVER 0

#if SERVER
/* FOR RUNNING ON LAB MACHINE WHERE TBB LIB VERSION IS OLDER */
#include <tbb/pipeline.h>
#define FILTER_PARALLEL tbb::filter::parallel
#define FILTER_SERIAL tbb::filter::serial_in_order
#define FLOW_TYPE tbb::flow_control
#define MAX_THREAD_COUNT 96
#else
#include <tbb/tbb.h>
#define FILTER_PARALLEL filter_mode::parallel
#define FILTER_SERIAL filter_mode::serial_in_order
#define FLOW_TYPE detail::d1::flow_control
#define MAX_THREAD_COUNT 8
#endif

using namespace tbb;

extern "C" {
#include "filter.h"
#include "pipeline.h"
}

enum OP{
	OP_SCALE,
	OP_DESATURATE,
	OP_HOR_FLIP,
	OP_SOBEL,
};

class pipelineInput{
public:
    pipelineInput(image_dir_t* image_dir){
        this->image_dir = image_dir;
    }

    image_t * operator()(FLOW_TYPE &flow) const {
        image_t *input_image = image_dir_load_next(this->image_dir);
        if(input_image) return input_image;
        flow.stop();
        return NULL;
    }

private:
    image_dir_t* image_dir;
};

class PipelineCompute{
public:
    PipelineCompute(enum OP operation): operation(operation) {}

    image_t * operator()(image_t* input_image)const {
        image_t* output_image = NULL;
        switch (this->operation){
            case OP_SCALE:
                output_image = filter_scale_up(input_image, 2);
                break;
            case OP_DESATURATE:
                output_image = filter_desaturate(input_image);
                break;
            case OP_HOR_FLIP:
                output_image = filter_horizontal_flip(input_image);
                break;
            case OP_SOBEL:
                output_image = filter_sobel(input_image);
                break;
        }
        image_destroy(input_image);
        return output_image;
    }

private:
    const enum OP operation;
};

class PipelineOutput{
public:
    PipelineOutput(image_dir_t* image_dir){
        this->image_dir = image_dir;
    }

    void operator()(image_t *input_image) const {
        if(!input_image) return;
        image_dir_save(this->image_dir, input_image);
        printf(".");
		fflush(stdout);
        image_destroy(input_image);
    }

private:
    image_dir_t* image_dir;
};

int pipeline_tbb(image_dir_t* image_dir) {
    parallel_pipeline(
        MAX_THREAD_COUNT,
        make_filter<void, image_t *>(FILTER_SERIAL, pipelineInput(image_dir)) &
        make_filter<image_t *, image_t *>(FILTER_PARALLEL, PipelineCompute(OP_SCALE)) &
        make_filter<image_t *, image_t *>(FILTER_PARALLEL, PipelineCompute(OP_DESATURATE)) &
        make_filter<image_t *, image_t *>(FILTER_PARALLEL, PipelineCompute(OP_HOR_FLIP)) &
        make_filter<image_t *, image_t *>(FILTER_PARALLEL, PipelineCompute(OP_SOBEL)) &
        make_filter<image_t *, void>(FILTER_PARALLEL, PipelineOutput(image_dir))
    );
    return 0;
}
