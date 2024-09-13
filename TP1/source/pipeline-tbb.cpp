#include <iostream>

#define SERVER_RUN 0

#if SERVER_RUN
/* FOR RUNNING ON LAB MACHINE WHERE TBB LIB VERSION IS OLDER */
#include <tbb/pipeline.h>
#define FILTER_PARALLEL tbb::filter::parallel
#define FILTER_SERIAL tbb::filter::serial_in_order
#else
#include <tbb/tbb.h>
#define FILTER_PARALLEL filter_mode::parallel
#define FILTER_SERIAL filter_mode::serial_in_order
#endif

extern "C" {
#include "filter.h"
#include "pipeline.h"
}

using namespace tbb;

#define DEFAULT_THREAD_COUNT 16

enum OP{
	OP_SCALE,
	OP_DESATURATE,
	OP_HOR_FLIP,
	OP_EDGE_DETECT,
};

class PipelineInput{
public:
    PipelineInput(image_dir_t* image_dir){
        this->image_dir = image_dir;
    }

    image_t * operator()(detail::d1::flow_control &flow) const {
        image_t *img = image_dir_load_next(this->image_dir);
        if(img) return img;
        flow.stop();
        return NULL;
    }

private:
    image_dir_t* image_dir;
};

class PipelineCompute{
public:
    PipelineCompute(enum OP operation): operation(operation) {}

    image_t * operator()(image_t *input) const {
        image_t *output = NULL;
        switch (this->operation){
		case OP_SCALE:
			output = filter_scale_up(input, 2);
            break;
		case OP_DESATURATE:
			output = filter_desaturate(input);
            break;
		case OP_HOR_FLIP:
			output = filter_horizontal_flip(input);
            break;
		case OP_EDGE_DETECT:
			output = filter_sobel(input);
            break;
		}
        image_destroy(input);
        return output;
    }

private:
    const enum OP operation; 
};

class PipelineOutput{
public:
    PipelineOutput(image_dir_t* image_dir){
        this->image_dir = image_dir;
    }

    void operator()(image_t *input) const {
        if(!input) return;
		image_dir_save(this->image_dir, input);
        std::cout << "OUTPUT IMAGE: " << input->id << std::endl;
        image_destroy(input);
    }

private:
    image_dir_t* image_dir;
};

static int processor_count = -1;

int pipeline_tbb(image_dir_t* image_dir) {
    if(processor_count == -1){
        processor_count = std::thread::hardware_concurrency();
        std::cout << "PROCESSOR COUNT: " << processor_count << std::endl;
    }
    
    parallel_pipeline(
        (processor_count == 0) ? DEFAULT_THREAD_COUNT : processor_count,
        make_filter<void, image_t *>(FILTER_SERIAL, PipelineInput(image_dir))      &
        make_filter<image_t *, image_t *>(FILTER_PARALLEL, PipelineCompute(OP_SCALE))       &
        make_filter<image_t *, image_t *>(FILTER_PARALLEL, PipelineCompute(OP_DESATURATE))  &
        make_filter<image_t *, image_t *>(FILTER_PARALLEL, PipelineCompute(OP_HOR_FLIP))    &
        make_filter<image_t *, image_t *>(FILTER_PARALLEL, PipelineCompute(OP_EDGE_DETECT)) &
        make_filter<image_t *, void>(FILTER_PARALLEL, PipelineOutput(image_dir))            
    );

    return 0;
}
