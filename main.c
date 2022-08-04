#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define constants
const int WINDOW = 3;
const int WINDOW_ALL = (WINDOW*2+1) * (WINDOW*2+1);

// Compare function for quick sort
int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

// Create 2D image data from 3D data to speed up operations
// x, y, z = 3D points, must have same shape
// count = output image that contains number of pixels that were averaged together
// image = output final image
// n = total length of x == y == z points
// dims = dimensions of image
void createImage(ssize_t *x, ssize_t *y, ssize_t *z, ssize_t *count, ssize_t *image, ssize_t n, ssize_t *dims)
{
    ssize_t i, j, w, M, N, y_index, x_index;

    M = dims[0]; N = dims[1];

    // For each x, y, z value
    for (i=0; i < n; i++) {
        // Get x and y coordinates by current pixel position
        y_index = y[i];
        x_index = x[i];

        // Get z value by x and y coordinates and apply dynamic averaging if more pixels are on the same position
        image[y_index*N + x_index] = (image[y_index*N + x_index] * count[y_index*N + x_index] + z[i]) / (count[y_index*N + x_index] + 1);
        count[y_index*N + x_index] = count[y_index*N + x_index] + 1;

        // Deprecated: The same thing without dynamic averaging
        // image[y_index*N + x_index] = z[i];
    }
}

// Median image just for zero pixels
//   => bring as much information as possible to pixels where is no information (or just few points from LiDAR)
// x, y, z = 3D points, must have same shape
// image = input from function createImage
// median = output image
// dims = dimensions of image
void medianImage(ssize_t *image, ssize_t *median, ssize_t *dims) {
    ssize_t i, j, M, N;
    int k, l, lastIndex;

    M = dims[0]; N = dims[1];

    // For each pixel without padding due to window size
    for (i = WINDOW; i < M - WINDOW; i++) {
        for (j = WINDOW; j < N - WINDOW; j++) {
            // Apply median just for 0 pixels to prevent information distortion
            if (image[i * N + j] == 0) {
                int medianArray[WINDOW_ALL] = {0};
                int lastIndex = 0;

                // For each pixel get values from neighbour pixels (window)
                for (k = -WINDOW; k <= WINDOW; k++) {
                    for (l = -WINDOW; l <= WINDOW; l++) {
                        // If pixel value is 0 => skip
                        if (image[(i + k) * N + (j + l)] != 0) {
                            medianArray[lastIndex] = image[(i + k) * N + (j + l)];
                            lastIndex++;
                        }
                    }
                }

                // If there are some data
                if (lastIndex != 0) {
                    // Sort pixel values from window to get median value
                    qsort (medianArray, lastIndex, sizeof(int), compare);
                    median[i * N + j] = (ssize_t) medianArray[(int) round(lastIndex / 2)];
                }
            }
        }
    }
}
