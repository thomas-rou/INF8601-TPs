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
    MPI_Status status;
    // Create a 2D cartesian communicator
    if (MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]){dim_x, dim_y}, (int[]){1, 1}, 0, &heatsim->communicator) != MPI_SUCCESS) {
        LOG_ERROR("MPI_Cart_create failed");
        goto fail_exit;
    }

    // Get the rank of the current communicator
    if(MPI_Comm_rank(heatsim->communicator, &heatsim->rank) == MPI_ERR_COMM) {
        LOG_ERROR("MPI_Comm_rank failed");
        goto fail_exit;
    }

    // Get the number of processes in the communicator
    if(MPI_Comm_size(heatsim->communicator, &heatsim->rank_count) != MPI_SUCCESS) {
        LOG_ERROR("MPI_Comm_size failed");
        goto fail_exit;
    }

    // Get the coord of the current process
    if(MPI_Cart_coords(heatsim->communicator, heatsim->rank, 2, heatsim->coordinates) != MPI_SUCCESS) {
        LOG_ERROR("MPI_Cart_coords failed");
        goto fail_exit;
    }

    // Get the rank of the neighbors communicators
    // Get north and south neighbors
    if(MPI_Cart_shift(heatsim->communicator, 0, 1, &heatsim->rank_north_peer, &heatsim->rank_south_peer) != MPI_SUCCESS) {
        LOG_ERROR("MPI_Cart_shift failed");
        goto fail_exit;
    }

    // Get west and east neighbors
    if(MPI_Cart_shift(heatsim->communicator, 1, 1, &heatsim->rank_west_peer, &heatsim->rank_east_peer) != MPI_SUCCESS) {
        LOG_ERROR("MPI_Cart_shift failed");
        goto fail_exit;
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
    for (int destination_rank = 1; destination_rank < heatsim->rank_count; destination_rank++) {
        grid_t* grid = cart2d_get_grid(cart, heatsim->coordinates[0], heatsim->coordinates[1]);
        if (grid == NULL) {
            LOG_ERROR("failed to get grid at (%d, %d) (rank 0)", heatsim->coordinates[0],
                      heatsim->coordinates[1]);
            goto fail_exit;
        }

        MPI_Request request;
        MPI_Status status;

        // Send the width, height and padding of the grid using pointer to width (struct have height and padding after width)
        if(MPI_Isend(&grid->width, 3, MPI_UNSIGNED, destination_rank, 0, heatsim->communicator, &request) != MPI_SUCCESS) {
            LOG_ERROR("MPI_Isend failed to send grid info to rank %d", destination_rank);
            goto fail_exit;
        }

        if(MPI_Wait(&request, &status) != MPI_SUCCESS) {
            LOG_ERROR("MPI_Wait failed to wait for grid info to rank %d", destination_rank);
            goto fail_exit;
        }

        // Create a sendable datatype for the grid
        MPI_Datatype ElementType[] = {MPI_DOUBLE};
        int elementBlockSize[] = {grid->width_padded * grid->height_padded};
        MPI_Aint elementDisplacement[] = {0};
        MPI_Datatype datatype;

        // Create the datatype
        if(MPI_Type_create_struct(1, elementBlockSize, elementDisplacement, ElementType, &datatype) != MPI_SUCCESS) {
            LOG_ERROR("MPI_Type_create_struct failed to create datatype for grid");
            goto fail_exit;
        }

        // Commit the datatype to allow it to be used in MPI_Isend
        if(MPI_Type_commit(&datatype) != MPI_SUCCESS) {
            LOG_ERROR("MPI_Type_commit failed to commit datatype for grid");
            goto fail_exit;
        }

        // Send the grid data with the custom type
        if(MPI_Isend(grid->data, 1, datatype, destination_rank, 0, heatsim->communicator, &request) != MPI_SUCCESS) {
            LOG_ERROR("MPI_Isend failed to send grid data to rank %d", destination_rank);
            goto fail_exit;
        }

        if(MPI_Wait(&request, &status) != MPI_SUCCESS) {
            LOG_ERROR("MPI_Wait failed to wait for grid data to rank %d", destination_rank);
            goto fail_exit;
        }


        // Free the datatype
        if(MPI_Type_free(&datatype) != MPI_SUCCESS) {
            LOG_ERROR("MPI_Type_free failed to free datatype for grid");
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

fail_exit:
    return NULL;
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

fail_exit:
    return -1;
}

int heatsim_send_result(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 0);

    /*
     * TODO: Envoyer les données (`data`) du `grid` résultant au rang 0. Le
     *       `grid` n'a aucun rembourage (padding = 0);
     */

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

fail_exit:
    return -1;
}
