#include <stdio.h>
#include <pthread.h>

#include "filter.h"
#include "pipeline.h"
#include "queue.h"

#define SERVER 0

#if SERVER
#define NUM_PARALLEL_PIPELINES 20
#else
#define NUM_PARALLEL_PIPELINES 2
#endif

#define NUM_PIPELINE_STEPS 4
#define QUEUE_SIZE 32

enum OP{
	OP_SCALE,
	OP_DESATURATE,
	OP_HOR_FLIP,
	OP_SOBEL,
};

struct pipeline_input_args{
	image_dir_t* image_dir;
	struct queue *output_queue;
	unsigned int nbr_parallel_pipelines;
};

struct pipeline_ops_args{
	struct queue *input_queue;
	struct queue *output_queue;
	enum OP operation_type;
};

struct pipeline_output_args{
	struct queue *input_queue;
	image_dir_t* image_dir;
};

void * pipeline_input_callback (void *thread_args){
	struct pipeline_input_args *input_args = (struct pipeline_input_args *)thread_args;
	image_t *input_image = NULL;
	while((input_image = image_dir_load_next(input_args->image_dir)) != NULL){
		queue_push(input_args->output_queue, input_image);
	}
	// Send enough NULL messages to end all parallels pipelines/threads
	for(unsigned int i = 0; i < input_args->nbr_parallel_pipelines; i++){
		queue_push(input_args->output_queue, NULL);
	}
	return NULL;
}

void * pipeline_operations_callback (void *thread_args){
	struct pipeline_ops_args *operation = (struct pipeline_ops_args *)thread_args;
	image_t *input_image = NULL;
	while((input_image = (image_t *)queue_pop(operation->input_queue)) != NULL){
		image_t *output_image = NULL;
		switch (operation->operation_type){
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
		// Error if processing step fails
		if(output_image == NULL){
			printf("ERROR IN IMG PROCESSING STEP: %d\n", operation->operation_type);
			continue;
		}
		queue_push(operation->output_queue, output_image);
	}
	// once the piepline is done --> send a signal to the next step to exit
	queue_push(operation->output_queue, NULL);
	return NULL;
}

void * pipeline_output_callback (void *thread_args){
	struct pipeline_output_args *output_args = (struct pipeline_output_args *)thread_args;
	image_t *input_image = NULL;
	while((input_image = (image_t *)queue_pop(output_args->input_queue)) != NULL){
		int status = image_dir_save(output_args->image_dir, input_image);
		printf(".");
		fflush(stdout);
		image_destroy(input_image);
	}
	return NULL;
}

int pipeline_pthread(image_dir_t* image_dir) {
	pthread_t op_threads[NUM_PARALLEL_PIPELINES][NUM_PIPELINE_STEPS];
	pthread_t input_thread;
	pthread_t output_threads[NUM_PARALLEL_PIPELINES];

	// Create struct shared by each operation of the same step
	struct pipeline_ops_args operations[NUM_PIPELINE_STEPS];
	queue_t *queues[NUM_PIPELINE_STEPS +1];

	// Create queues
	for(unsigned int i = 0; i < NUM_PIPELINE_STEPS + 1; i++){
		queues[i] = queue_create(QUEUE_SIZE);
	}

	// Create compute threads
	for(unsigned int i = 0; i < NUM_PIPELINE_STEPS; i++){
		operations[i] = (struct pipeline_ops_args){.input_queue = queues[i], .output_queue = queues[i + 1], .operation_type = (enum OP)(i)};
		for (unsigned int j = 0; j < NUM_PARALLEL_PIPELINES; j++){
			pthread_create(&op_threads[j][i], NULL, pipeline_operations_callback, &operations[i]);
		}
	}

	// Create input threads
	struct pipeline_input_args input_args = {.image_dir = image_dir, .nbr_parallel_pipelines = NUM_PARALLEL_PIPELINES, .output_queue = queues[0]};
	pthread_create(&input_thread, NULL, pipeline_input_callback, &input_args);

	// Create output threads
	struct pipeline_output_args output_args[NUM_PARALLEL_PIPELINES];
	for(unsigned int i = 0; i < NUM_PARALLEL_PIPELINES; i++){
		output_args[i] = (struct pipeline_output_args){.input_queue = queues[NUM_PIPELINE_STEPS], .image_dir = image_dir};
		pthread_create(&output_threads[i], NULL, pipeline_output_callback, &output_args);
	}

	// wait for end of all threads
	pthread_join(input_thread, NULL);
	for(unsigned int i = 0; i < NUM_PIPELINE_STEPS; i++){
		for(unsigned int j = 0; j < NUM_PARALLEL_PIPELINES; j++){
			pthread_join(op_threads[j][i], NULL);
		}
	}
	for(unsigned int i = 0; i < NUM_PARALLEL_PIPELINES; i++){
		pthread_join(output_threads[i], NULL);
	}

	// Destroy all queues
	for(unsigned int i = 0; i <= NUM_PIPELINE_STEPS; i++){
		queue_destroy(queues[i]);
	}

	return 0;
}
