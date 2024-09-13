#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#include "filter.h"
#include "pipeline.h"
#include "queue.h"

#define NUM_PIPELINE_STEPS 4
#define DEFAULT_NUM_PARALLEL_PIPELINES 2
#define QUEUE_SIZE 32

enum OP{
	OP_SCALE,
	OP_DESATURATE,
	OP_HOR_FLIP,
	OP_EDGE_DETECT,
};

struct pipeline_input_args{
	struct queue *output;
	image_dir_t *img_dir;
	unsigned int parallel_pipelines;
};

struct img_op_args{
	struct queue *input;
	struct queue *output;
	enum OP operation;
	unsigned int parallel_pipelines;
};

struct pipeline_output_args{
	struct queue *input;
	image_dir_t *img_dir;
	unsigned int count;
};

void * pipeline_input_callback(void *thread_args){
	struct pipeline_input_args *args = (struct pipeline_input_args *)thread_args;
	image_t *input = NULL;
	while((input = image_dir_load_next(args->img_dir)) != NULL){
		queue_push(args->output, input);
	}
	/* SEND AS MANY NULLS AS THERE ARE THREADS IN THE NEXT STEP */
	for(unsigned int i = 0; i < args->parallel_pipelines; i++)
		queue_push(args->output, NULL);
	return NULL;
}

void * img_op_callback(void *thread_args){
	struct img_op_args *args = (struct img_op_args *)thread_args;
	image_t *input = NULL;
	while((input = (image_t *)queue_pop(args->input)) != NULL){
		image_t *output = NULL;
		switch (args->operation){
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
		/* IN CASE IMAGE PROCESSING STEP FAILS */
		if(output == NULL){
			printf("ERROR IN IMG PROCESSING STEP: %d\n", args->operation);
			continue;
		} 
		queue_push(args->output, output);
	}
	/* TO ENABLE NEXT PIPELINE THREADS TO EXIT WHILE LOOP */
	for(unsigned int i = 0; i < args->parallel_pipelines; i++)
		queue_push(args->output, NULL);
	return NULL;
}

void * pipeline_output_callback(void *thread_args){
	struct pipeline_output_args *args = (struct pipeline_output_args *)thread_args;
	image_t *input = NULL;
	while((input = (image_t *)queue_pop(args->input)) != NULL){
		/* SHOULD NOT BE A PROBLEM TO ACCESS READ-ONLY img_dir AT THE SAME TIME AS PIPELINE INPUT THREAD */
		int status = image_dir_save(args->img_dir, input);
		args->count++;
		image_destroy(input);
	}
	return NULL;
}

static int processor_count = -1;

int pipeline_pthread(image_dir_t* image_dir) {
	/* 12 PTHREAD */
	/* 8 TBB */

	if(processor_count == -1){
		processor_count = ((sysconf(_SC_NPROCESSORS_ONLN) - 1) / NUM_PIPELINE_STEPS) + 1;
		printf("NUMBER OF PARALLEL PIPELINES: %d\n", processor_count);
	}

	const unsigned int parallel_pipelines = (processor_count <= 0) ? DEFAULT_NUM_PARALLEL_PIPELINES : processor_count;
	pthread_t threads[parallel_pipelines][NUM_PIPELINE_STEPS];
	/* ALL PARALLEL THREADS OF A SAME STEP CAN SHARE THE SAME ARG STRUCT */
	struct img_op_args args[NUM_PIPELINE_STEPS];
	queue_t *queues[NUM_PIPELINE_STEPS + 1];

	/* INIT QUEUES */
	for(unsigned int i = 0; i < NUM_PIPELINE_STEPS + 1; i++){
		queues[i] = queue_create(QUEUE_SIZE);
	}

	/* INIT COMPUTE THREADS */
	for(unsigned int i = 0; i < NUM_PIPELINE_STEPS; i++){
		args[i] = (struct img_op_args){.input = queues[i] , .output = queues[i + 1], .operation = (enum OP)(OP_SCALE + i), .parallel_pipelines = parallel_pipelines};
		for(unsigned int j = 0; j < parallel_pipelines; j++)
			pthread_create(&threads[j][i], NULL, img_op_callback, &args[i]);
	}

	/* INIT PIPELINE INPUT THREAD */
	pthread_t input_thread;
	struct pipeline_input_args input_args = {.output = queues[0], .img_dir = image_dir, .parallel_pipelines = parallel_pipelines};
	pthread_create(&input_thread, NULL, pipeline_input_callback, &input_args);

	/* INIT PIPELINE OUTPUT THREAD */
	pthread_t output_threads[parallel_pipelines];
	struct pipeline_output_args output_args[parallel_pipelines];
	for(unsigned int i = 0; i < parallel_pipelines; i++){
		output_args[i] = (struct pipeline_output_args){.input = queues[NUM_PIPELINE_STEPS], .img_dir = image_dir, .count = 0};
		pthread_create(&output_threads[i], NULL, pipeline_output_callback, &output_args[i]);
	}

	/* WAIT FOR END AND CLEANUP */
	pthread_join(input_thread, NULL);
	for(unsigned int i = 0; i < parallel_pipelines; i++)
		pthread_join(output_threads[i], NULL);
	for(unsigned int i = 0; i < NUM_PIPELINE_STEPS; i++){
		for(unsigned int j = 0; j < parallel_pipelines; j++)
			pthread_join(threads[j][i], NULL);
	}
	for(unsigned int i = 0; i < NUM_PIPELINE_STEPS + 1; i++)
		queue_destroy(queues[i]);

	return 0;
}
