#include <assert.h>
#include <stddef.h>

#include "heatsim.h"
#include "log.h"

int heatsim_init(heatsim_t* heatsim, unsigned int dim_x, unsigned int dim_y) {
    /*
     * TODO: Initialiser tous les membres de la structure `heatsim`.
     *       Le communicateur doit être périodique. Le communicateur
     *       cartésien est périodique en X et Y.
     */
    if(MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]){dim_y, dim_x}, (int[]){1, 1}, 0, &heatsim->communicator) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT INIT CART");
        goto fail_exit;
    } 
    if(MPI_Comm_rank(heatsim->communicator, &heatsim->rank) == MPI_ERR_COMM){
        LOG_ERROR("COULD NOT GET RANK");
        goto fail_exit;
    } 
    if(MPI_Comm_size(heatsim->communicator, &heatsim->rank_count) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT GET RANK SIZE");
        goto fail_exit;
    } 
    if(MPI_Cart_coords(heatsim->communicator, heatsim->rank, 2, heatsim->coordinates) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT GET RANK COORDS");
        goto fail_exit;
    } 

    int status = 0;

    int north_coords[] = {heatsim->coordinates[0] - 1, heatsim->coordinates[1]};
    status |= MPI_Cart_rank(heatsim->communicator, north_coords, &heatsim->rank_north_peer);

    int south_coords[] = {heatsim->coordinates[0] + 1, heatsim->coordinates[1]};
    status |= MPI_Cart_rank(heatsim->communicator, south_coords, &heatsim->rank_south_peer);

    int west_coords[] = {heatsim->coordinates[0], heatsim->coordinates[1] - 1};
    status |= MPI_Cart_rank(heatsim->communicator, west_coords, &heatsim->rank_west_peer);
    
    int east_coords[] = {heatsim->coordinates[0], heatsim->coordinates[1] + 1};
    status |= MPI_Cart_rank(heatsim->communicator, east_coords, &heatsim->rank_east_peer);
    
    if(status != MPI_SUCCESS){
        LOG_ERROR("COULD NOT LOAD THE RANK FROM THE COORDINATE FOR NEIGHBOURS");
        goto fail_exit;
    }

    {
        const int tmp = heatsim->coordinates[0];
        heatsim->coordinates[0] = heatsim->coordinates[1];
        heatsim->coordinates[1] = tmp;
    }

    return 0;

fail_exit:
    return -1;
}

int heatsim_send_grids(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Envoyer toutes les `grid` aux autres rangs. Cette fonction
     *       est appelé pour le rang 0. Par exemple, si le rang 3 est à la
     *       coordonnée cartésienne (0, 2), alors on y envoit le `grid`
     *       à la position (0, 2) dans `cart`.
     *
     *       Il est recommandé d'envoyer les paramètres `width`, `height`
     *       et `padding` avant les données. De cette manière, le receveur
     *       peut allouer la structure avec `grid_create` directement.
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée.
     */
    for(int i = 1; i < heatsim->rank_count; i++){
        int coords[2];
        if(MPI_Cart_coords(heatsim->communicator, i, 2, coords) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT GET CARTESIAN CORDINATES");
            goto fail_exit;
        }

        grid_t *grid = cart2d_get_grid(cart, coords[1], coords[0]);
        MPI_Request request;
        MPI_Status status;
        
        if(MPI_Isend(&grid->width, 3, MPI_UNSIGNED, i, 0, heatsim->communicator, &request) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT SEND GRID INFO TO RANK: %d", i);
            goto fail_exit;
        }

        if(MPI_Wait(&request, &status) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT WAIT FOR SENDING GRID INFO TO RANK: %d", i);
            goto fail_exit;
        }

        MPI_Datatype ElementFieldTypes[] = {MPI_DOUBLE};
        int ElementBlockLength[] = {grid->width_padded * grid->height_padded};
        MPI_Aint ElementOffset[] = {0};
        MPI_Datatype datatype;
        if(MPI_Type_create_struct(1, ElementBlockLength, ElementOffset, ElementFieldTypes, &datatype) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT ADD GRID STRUCT");
            goto fail_exit;
        }

        if(MPI_Type_commit(&datatype) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT COMMIT GRID STRUCT");
            goto fail_exit;
        }

        if(MPI_Isend(grid->data, 1, datatype, i, 1, heatsim->communicator, &request) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT SEND GRID STRUCT TO RANK: %d", i);
            goto fail_exit;
        }

        if(MPI_Wait(&request, &status) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT WAIT FOR SEND GRID STRUCT TO RANK: %d", i);
            goto fail_exit;
        }

        if(MPI_Type_free(&datatype) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT FREE GRID STRUCT TYPE");
            goto fail_exit;
        }
    }

    return 0;

fail_exit:
    return -1;
}

grid_t* heatsim_receive_grid(heatsim_t* heatsim) {
    /*
     * TODO: Recevoir un `grid ` du rang 0. Il est important de noté que
     *       toutes les `grid` ne sont pas nécessairement de la même
     *       dimension (habituellement ±1 en largeur et hauteur). Utilisez
     *       la fonction `grid_create` pour allouer un `grid`.
     *
     *       Utilisez `grid_create` pour allouer le `grid` à retourner.
     */

    unsigned int dim_buffer[3];
    MPI_Request request;
    MPI_Status status;

    if(MPI_Irecv(dim_buffer, 3, MPI_UNSIGNED, 0, 0, heatsim->communicator, &request) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT ASYNC REVEIVE GRID INFO");
        goto fail_exit;
    }

    if(MPI_Wait(&request, &status) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT WAIT FOR ASYNC REVEIVE GRID INFO");
        goto fail_exit;
    }

    grid_t *grid = grid_create(dim_buffer[0], dim_buffer[1], dim_buffer[2]);

    MPI_Datatype ElementFieldTypes[] = {MPI_DOUBLE};
    int ElementBlockLength[] = {grid->width_padded * grid->height_padded};
    MPI_Aint ElementOffset[] = {0};
    MPI_Datatype datatype;

    if(MPI_Type_create_struct(1, ElementBlockLength, ElementOffset, ElementFieldTypes, &datatype) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT INSTANTIATE GRID");
        goto fail_exit;
    }

    if(MPI_Type_commit(&datatype) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT COMMIT GRID DATATYPE");
        goto fail_exit;
    }

    if(MPI_Irecv(grid->data, 1, datatype, 0, 1, heatsim->communicator, &request) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT ASYNC RECEIVE GRID DATA");
        goto fail_exit;
    }

    if(MPI_Wait(&request, &status) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT WAIT FOR RECEIVE GRID DATA");
        goto fail_exit;
    }

    if(MPI_Type_free(&datatype) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT FREE GRID DATA DATATYPE");
        goto fail_exit;
    }

    return grid;

fail_exit:
    return NULL;
}

#define HEATSIM_EXCHANGE_BORDERS_ASYNC 0
#if HEATSIM_EXCHANGE_BORDERS_ASYNC
int heatsim_exchange_borders_async(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 1);

    /*
     * TODO: Échange les bordures de `grid`, excluant le rembourrage, dans le
     *       rembourrage du voisin de ce rang. Par exemple, soit la `grid`
     *       4x4 suivante,
     *
     *                            +-------------+
     *                            | x x x x x x |
     *                            | x A B C D x |
     *                            | x E F G H x |
     *                            | x I J K L x |
     *                            | x M N O P x |
     *                            | x x x x x x |
     *                            +-------------+
     *
     *       où `x` est le rembourrage (padding = 1). Ce rang devrait envoyer
     *
     *        - la bordure [A B C D] au rang nord,
     *        - la bordure [M N O P] au rang sud,
     *        - la bordure [A E I M] au rang ouest et
     *        - la bordure [D H L P] au rang est.
     *
     *       Ce rang devrait aussi recevoir dans son rembourrage
     *
     *        - la bordure [A B C D] du rang sud,
     *        - la bordure [M N O P] du rang nord,
     *        - la bordure [A E I M] du rang est et
     *        - la bordure [D H L P] du rang ouest.
     *
     *       Après l'échange, le `grid` devrait avoir ces données dans son
     *       rembourrage provenant des voisins:
     *
     *                            +-------------+
     *                            | x m n o p x |
     *                            | d A B C D a |
     *                            | h E F G H e |
     *                            | l I J K L i |
     *                            | p M N O P m |
     *                            | x a b c d x |
     *                            +-------------+
     *
     *       Utilisez `grid_get_cell` pour obtenir un pointeur vers une cellule.
     */

    MPI_Request send_requests[4] = {NULL, NULL, NULL, NULL};

    /* HANDLE NORTH BORDER SEND */
    if(heatsim->rank != heatsim->rank_north_peer){
        int status = MPI_Isend(grid_get_cell(grid, 0, 0), grid->width, MPI_DOUBLE, 
            heatsim->rank_north_peer, 0, heatsim->communicator, &send_requests[0]);
        if(status != MPI_SUCCESS){
            LOG_ERROR("COULD NOT SEND NORTH BORDER FROM %d to %d", heatsim->rank, heatsim->rank_north_peer);
            goto fail_exit;
        }
    }else{
        memcpy(grid_get_cell_padded(grid, 1, grid->height_padded - 1), 
            grid_get_cell(grid, 0, 0), grid->width * sizeof(double));
    }

    /* HANDLE SOUTH BORDER SEND */
    if(heatsim->rank != heatsim->rank_south_peer){
        int status = MPI_Isend(grid_get_cell(grid, 0, grid->height - 1), grid->width, MPI_DOUBLE, 
            heatsim->rank_south_peer, 1, heatsim->communicator, &send_requests[1]);
        if(status != MPI_SUCCESS){
            LOG_ERROR("COULD NOT SEND SOUTH BORDER FROM %d to %d", heatsim->rank, heatsim->rank_south_peer);
            goto fail_exit;
        }
    }else{
        memcpy(grid_get_cell_padded(grid, 1, 0), 
            grid_get_cell(grid, 0, grid->height - 1), grid->width * sizeof(double));
    }

    MPI_Datatype datatype;
    MPI_Type_vector(grid->height, 1, grid->width_padded, MPI_DOUBLE, &datatype);
    MPI_Type_commit(&datatype);

    /* HANDLE EAST BORDER SEND */
    if(heatsim->rank != heatsim->rank_east_peer){
        int status = MPI_Isend(grid_get_cell(grid, grid->width - 1, 0), 1, datatype, heatsim->rank_east_peer, 
            0, heatsim->communicator, &send_requests[2]);
        if(status != MPI_SUCCESS){
            LOG_ERROR("COULD NOT SEND EAST BORDER FROM %d to %d", heatsim->rank, heatsim->rank_east_peer);
            goto fail_exit;
        }
    }else{
        for(int i = 0; i < grid->height; i++)
            *grid_get_cell_padded(grid, 0, i + 1) = *grid_get_cell(grid, grid->width - 1, i);
    }

    /* HANDLE WEST BORDER SEND */
    if(heatsim->rank != heatsim->rank_west_peer){
        int status = MPI_Isend(grid_get_cell(grid, 0, 0), 1, datatype, heatsim->rank_west_peer, 
            1, heatsim->communicator, &send_requests[3]);
        if(status != MPI_SUCCESS){
            LOG_ERROR("COULD NOT SEND WEST BORDER FROM %d to %d", heatsim->rank, heatsim->rank_west_peer);
            goto fail_exit;
        }
    }else{
        for(int i = 0; i < grid->height; i++)
            *grid_get_cell_padded(grid, grid->width_padded - 1, i + 1) = *grid_get_cell(grid, 0, i);
    }


    MPI_Request request;
    MPI_Status status;

    /* RECEIVE NORTH BORDERS */
    if(heatsim->rank != heatsim->rank_north_peer){
        MPI_Irecv(grid_get_cell_padded(grid, 1, 0), grid->width, MPI_DOUBLE,
            heatsim->rank_north_peer, 1, heatsim->communicator, &request);
        MPI_Wait(&request, &status);
    }

    /* RECEIVE SOUTH BORDERS */
    if(heatsim->rank != heatsim->rank_south_peer){
        MPI_Irecv(grid_get_cell_padded(grid, 1, grid->height_padded - 1), grid->width, MPI_DOUBLE,
            heatsim->rank_south_peer, 0, heatsim->communicator, &request);
        MPI_Wait(&request, &status);
    }

    /* RECEIVE WEST BORDERS */
    if(heatsim->rank != heatsim->rank_west_peer){
        MPI_Irecv(grid_get_cell_padded(grid, 0, 1), 1, datatype, 
            heatsim->rank_west_peer, 0, heatsim->communicator, &request);
        MPI_Wait(&request, &status);
    }

    /* RECEIVE EAST BORDERS */
    if(heatsim->rank != heatsim->rank_east_peer){
        MPI_Irecv(grid_get_cell_padded(grid, grid->width_padded - 1, 1), 1, datatype, 
            heatsim->rank_east_peer, 1, heatsim->communicator, &request);
        MPI_Wait(&request, &status);
    }

    /* FINALIZE SEND REQUESTS */
    MPI_Status send_status;
    for(unsigned int i = 0; i < sizeof(send_requests) / sizeof(send_requests[0]); i++){
        if(!send_requests[i]) continue;
        MPI_Wait(&send_requests[i], &send_status);
    }

    MPI_Type_free(&datatype);

    return 0;

fail_exit:
    return -1;
}
#endif //HEATSIM_EXCHANGE_BORDERS_ASYNC

static int handle_east_west_exchange(heatsim_t* heatsim, grid_t* grid){
    MPI_Datatype datatype;
    MPI_Type_vector(grid->height, 1, grid->width_padded, MPI_DOUBLE, &datatype);
    MPI_Type_commit(&datatype);

    int status;
    MPI_Status req_status;

    if(heatsim->rank % 2 == 0){

        /* SEND EAST BORDER */
        if(heatsim->rank != heatsim->rank_east_peer){
            status = MPI_Send(grid_get_cell(grid, grid->width - 1, 0), 1, 
                datatype, heatsim->rank_east_peer, 0, heatsim->communicator);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT SEND EAST BORDER FOR PAIR NODE", status);
                goto fail_exit;
            }
        }else{
            for(int i = 0; i < grid->height; i++)
                *grid_get_cell_padded(grid, 0, i + 1) = *grid_get_cell(grid, grid->width - 1, i);
        }

        /* RECEIVE WEST BORDER */
        if(heatsim->rank != heatsim->rank_west_peer){
            status = MPI_Recv(grid_get_cell_padded(grid, 0, 1), 1, datatype, 
                heatsim->rank_west_peer, 0, heatsim->communicator, &req_status);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT RECEIVE WEST BORDER FOR PAIR NODE", status);
                goto fail_exit;
            }
        }

        /* SEND WEST BORDER */
        if(heatsim->rank != heatsim->rank_west_peer){
            status = MPI_Send(grid_get_cell(grid, 0, 0), 1, datatype, heatsim->rank_west_peer, 1, heatsim->communicator);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT SEND WEST BORDER FOR PAIR NODE", status);
                goto fail_exit;
            }
        }else{
            for(int i = 0; i < grid->height; i++)
                *grid_get_cell_padded(grid, grid->width_padded - 1, i + 1) = *grid_get_cell(grid, 0, i);
        }

        /* RECEIVE EAST BORDER */
        if(heatsim->rank != heatsim->rank_east_peer){
            status = MPI_Recv(grid_get_cell_padded(grid, grid->width_padded - 1, 1), 1, 
                datatype, heatsim->rank_east_peer, 1, heatsim->communicator, &req_status);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT RECEIVE EAST BORDER FOR PAIR NODE", status);
                goto fail_exit;
            }
        }

    }else{

        /* RECEIVE WEST BORDER */
        if(heatsim->rank != heatsim->rank_west_peer){
            status = MPI_Recv(grid_get_cell_padded(grid, 0, 1), 1, datatype, 
                heatsim->rank_west_peer, 0, heatsim->communicator, &req_status);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT RECEIVE WEST BORDER FOR PAIR NODE", status);
                goto fail_exit;
            }
        }

        /* SEND EAST BORDER */
        if(heatsim->rank != heatsim->rank_east_peer){
            status = MPI_Send(grid_get_cell(grid, grid->width - 1, 0), 1, 
                datatype, heatsim->rank_east_peer, 0, heatsim->communicator);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT SEND EAST BORDER FOR PAIR NODE", status);
                goto fail_exit;
            }
        }else{
            for(int i = 0; i < grid->height; i++)
                *grid_get_cell_padded(grid, 0, i + 1) = *grid_get_cell(grid, grid->width - 1, i);
        }

        /* RECEIVE EAST BORDER */
        if(heatsim->rank != heatsim->rank_east_peer){
            status = MPI_Recv(grid_get_cell_padded(grid, grid->width_padded - 1, 1), 1, 
                datatype, heatsim->rank_east_peer, 1, heatsim->communicator, &req_status);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT RECEIVE EAST BORDER FOR PAIR NODE", status);
                goto fail_exit;
            }
        }

        /* SEND WEST BORDER */
        if(heatsim->rank != heatsim->rank_west_peer){
            status = MPI_Send(grid_get_cell(grid, 0, 0), 1, datatype, heatsim->rank_west_peer, 1, heatsim->communicator);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT SEND WEST BORDER FOR PAIR NODE", status);
                goto fail_exit;
            }
        }else{
            for(int i = 0; i < grid->height; i++)
                *grid_get_cell_padded(grid, grid->width_padded - 1, i + 1) = *grid_get_cell(grid, 0, i);
        }

    }
    
    MPI_Type_free(&datatype);

    return 0;

fail_exit:
    return -1;
}

static int handle_north_south_exchange(heatsim_t *heatsim, grid_t *grid){
    int status;
    MPI_Status req_status;

    if(heatsim->coordinates[1] % 2 == 0){

        /* SEND NORTH BORDER */
        if(heatsim->rank != heatsim->rank_north_peer){
            status = MPI_Send(grid_get_cell(grid, 0, 0), grid->width, 
                MPI_DOUBLE, heatsim->rank_north_peer, 0, heatsim->communicator);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT SEND NORTH BORDER PAIR NODE", status);
                goto fail_exit;
            }
        }else{
            memcpy(grid_get_cell_padded(grid, 1, grid->height_padded - 1), 
                grid_get_cell(grid, 0, 0), grid->width * sizeof(double));
        }

        /* RECEIVE SOUTH BORDER */
        if(heatsim->rank != heatsim->rank_south_peer){
            status = MPI_Recv(grid_get_cell_padded(grid, 1, grid->height_padded - 1), 
                grid->width, MPI_DOUBLE, heatsim->rank_south_peer, 0, heatsim->communicator, &req_status);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT RECEIVE SOUTH BORDER PAIR NODE", status);
                goto fail_exit;
            }
        }

        /* SEND SOUTH BORDER */
        if(heatsim->rank != heatsim->rank_south_peer){
            status = MPI_Send(grid_get_cell(grid, 0, grid->height - 1), grid->width, 
                MPI_DOUBLE, heatsim->rank_south_peer, 1, heatsim->communicator);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT SEND SOUTH BORDER PAIR NODE", status);
                goto fail_exit;
            }
        }else{
            memcpy(grid_get_cell_padded(grid, 1, 0), 
                grid_get_cell(grid, 0, grid->height - 1), grid->width * sizeof(double));
        }

        /* RECEIVE NORTH BORDER */
        if(heatsim->rank != heatsim->rank_north_peer){
            status = MPI_Recv(grid_get_cell_padded(grid, 1, 0), grid->width, 
                MPI_DOUBLE, heatsim->rank_north_peer, 1, heatsim->communicator, &req_status);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT RECEIVE NORTH BORDER PAIR NODE", status);
                goto fail_exit;
            }
        }

    }else{

        /* RECEIVE SOUTH BORDER */
        if(heatsim->rank != heatsim->rank_south_peer){
            status = MPI_Recv(grid_get_cell_padded(grid, 1, grid->height_padded - 1), 
                grid->width, MPI_DOUBLE, heatsim->rank_south_peer, 0, heatsim->communicator, &req_status);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT RECEIVE SOUTH BORDER PAIR NODE", status);
                goto fail_exit;
            }
        }

        /* SEND NORTH BORDER */
        if(heatsim->rank != heatsim->rank_north_peer){
            status = MPI_Send(grid_get_cell(grid, 0, 0), grid->width, 
                MPI_DOUBLE, heatsim->rank_north_peer, 0, heatsim->communicator);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT SEND NORTH BORDER PAIR NODE", status);
                goto fail_exit;
            }
        }else{
            memcpy(grid_get_cell_padded(grid, 1, grid->height_padded - 1), 
                grid_get_cell(grid, 0, 0), grid->width * sizeof(double));
        }

        /* RECEIVE NORTH BORDER */
        if(heatsim->rank != heatsim->rank_north_peer){
            status = MPI_Recv(grid_get_cell_padded(grid, 1, 0), grid->width, 
                MPI_DOUBLE, heatsim->rank_north_peer, 1, heatsim->communicator, &req_status);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT RECEIVE NORTH BORDER PAIR NODE", status);
                goto fail_exit;
            }
        }

        /* SEND SOUTH BORDER */
        if(heatsim->rank != heatsim->rank_south_peer){
            status = MPI_Send(grid_get_cell(grid, 0, grid->height - 1), grid->width, 
                MPI_DOUBLE, heatsim->rank_south_peer, 1, heatsim->communicator);
            if(status != MPI_SUCCESS){
                LOG_ERROR_MPI("COULD NOT SEND SOUTH BORDER PAIR NODE", status);
                goto fail_exit;
            }
        }else{
            memcpy(grid_get_cell_padded(grid, 1, 0), 
                grid_get_cell(grid, 0, grid->height - 1), grid->width * sizeof(double));
        }

    }

    return 0;

fail_exit:
    return -1;
}

int heatsim_exchange_borders(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 1);

    /*
     * TODO: Échange les bordures de `grid`, excluant le rembourrage, dans le
     *       rembourrage du voisin de ce rang. Par exemple, soit la `grid`
     *       4x4 suivante,
     *
     *                            +-------------+
     *                            | x x x x x x |
     *                            | x A B C D x |
     *                            | x E F G H x |
     *                            | x I J K L x |
     *                            | x M N O P x |
     *                            | x x x x x x |
     *                            +-------------+
     *
     *       où `x` est le rembourrage (padding = 1). Ce rang devrait envoyer
     *
     *        - la bordure [A B C D] au rang nord,
     *        - la bordure [M N O P] au rang sud,
     *        - la bordure [A E I M] au rang ouest et
     *        - la bordure [D H L P] au rang est.
     *
     *       Ce rang devrait aussi recevoir dans son rembourrage
     *
     *        - la bordure [A B C D] du rang sud,
     *        - la bordure [M N O P] du rang nord,
     *        - la bordure [A E I M] du rang est et
     *        - la bordure [D H L P] du rang ouest.
     *
     *       Après l'échange, le `grid` devrait avoir ces données dans son
     *       rembourrage provenant des voisins:
     *
     *                            +-------------+
     *                            | x m n o p x |
     *                            | d A B C D a |
     *                            | h E F G H e |
     *                            | l I J K L i |
     *                            | p M N O P m |
     *                            | x a b c d x |
     *                            +-------------+
     *
     *       Utilisez `grid_get_cell` pour obtenir un pointeur vers une cellule.
     */

    if(handle_east_west_exchange(heatsim, grid) != 0) goto fail_exit;
    if(handle_north_south_exchange(heatsim, grid) != 0) goto fail_exit;

    return 0;

fail_exit:
    return -1;
}

int heatsim_send_result(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 0);

    /*
     * TODO: Envoyer les données (`data`) du `grid` résultant au rang 0. Le
     *       `grid` n'a aucun rembourage (padding = 0);
     */

    MPI_Request request;
    MPI_Status status;
    int err;

    if(MPI_Isend(grid->data, grid->height * grid->width, MPI_DOUBLE, 0, 0, heatsim->communicator, &request) != MPI_SUCCESS){
        LOG_ERROR("COULD NOT ASYNC SEND RESULT TO MASTER FROM RANK: %d", heatsim->rank);
        goto fail_exit;
    }

    err = MPI_Wait(&request, &status);
    if(err != MPI_SUCCESS){
        LOG_ERROR("COULD NOT WAIT FOR ASYNC SEND TO MASTER FROM RANK: %d", heatsim->rank);
        LOG_ERROR_MPI("", err);
        goto fail_exit;
    } 

    return 0;

fail_exit:
    return -1;
}

int heatsim_receive_results(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Recevoir toutes les `grid` des autres rangs. Aucune `grid`
     *       n'a de rembourage (padding = 0).
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée
     *       qui va recevoir le contenue (`data`) d'un autre noeud.
     */

    for(unsigned int i = 1; i < heatsim->rank_count; i++){
        int coords[2];
        int err;

        if(MPI_Cart_coords(heatsim->communicator, i, 2, coords) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT GET CARTESIAN COORDS FOR RANKS: %d", i);
            goto fail_exit;
        }
            
        grid_t *grid = cart2d_get_grid(cart, coords[1], coords[0]);
        MPI_Request request;
        MPI_Status status;

        if(MPI_Irecv(grid_get_cell(grid, 0, 0), grid->width * grid->height, MPI_DOUBLE, i, 0, heatsim->communicator, &request) != MPI_SUCCESS){
            LOG_ERROR("COULD NOT ASYNC RECEIVE GRID DATA");
            goto fail_exit;
        }
        err = MPI_Wait(&request, &status);
        if(err != MPI_SUCCESS){
            LOG_ERROR_MPI("COULD NOT WAIT FOR RECEIVE GRID DATA REQUEST", err);
            goto fail_exit;
        } 
    }

    return 0;

fail_exit:
    return -1;
}
