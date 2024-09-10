/* DO NOT EDIT THIS FILE */

#include <stdio.h>

#include "filter.h"
#include "pipeline.h"

int pipeline_serial(image_dir_t* image_dir) {
    while (1) {
        image_t* image1 = image_dir_load_next(image_dir);
        if (image1 == NULL) {
            break;
        }

        image_t* image2 = filter_scale_up(image1, 2);
        image_destroy(image1);
        if (image2 == NULL) {
            goto fail_exit;
        }

        image_t* image3 = filter_desaturate(image2);
        image_destroy(image2);
        if (image3 == NULL) {
            goto fail_exit;
        }

        image_t* image4 = filter_horizontal_flip(image3);
        image_destroy(image3);
        if (image4 == NULL) {
            goto fail_exit;
        }

        image_t* image5 = filter_sobel(image4);
        image_destroy(image4);
        if (image5 == NULL) {
            goto fail_exit;
        }

        image_dir_save(image_dir, image5);
        printf(".");
        fflush(stdout);
        image_destroy(image5);
    }

    printf("\n");
    return 0;

fail_exit:
    return -1;
}
