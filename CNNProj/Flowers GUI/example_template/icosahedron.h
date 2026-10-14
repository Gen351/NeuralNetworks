#pragma once

#include <vector>

#include "vec3.h"

std::vector<Vec3> getIcosahedronVertices() {
    return { 
        {.x = -0.15,   .y =  0.2427, .z =  0},      // 0
        {.x =  0.15,   .y =  0.2427, .z =  0},      // 1
        {.x = -0.15,   .y = -0.2427, .z =  0},      // 2
        {.x =  0.15,   .y = -0.2427, .z =  0},      // 3
        
        {.x =  0,      .y = -0.15,   .z =  0.2427}, // 4
        {.x =  0,      .y =  0.15,   .z =  0.2427}, // 5
        {.x =  0,      .y = -0.15,   .z = -0.2427}, // 6
        {.x =  0,      .y =  0.15,   .z = -0.2427}, // 7
        
        {.x =  0.2427, .y =  0,      .z = -0.15},   // 8
        {.x =  0.2427, .y =  0,      .z =  0.15},   // 9
        {.x = -0.2427, .y =  0,      .z = -0.15},   // 10
        {.x = -0.2427, .y =  0,      .z =  0.15}    // 11
    };
}

std::vector<std::vector<int>> getIcosahedronFaces() {
    return {
        // 5 faces around vertex 0
        {0, 11, 5}, 
        {0, 5, 1}, 
        {0, 1, 7}, 
        {0, 7, 10}, 
        {0, 10, 11},
        
        // 10 adjacent faces connecting the upper and lower "rings"
        {1, 5, 9}, 
        {5, 11, 4}, 
        {11, 10, 2}, 
        {10, 7, 6}, 
        {7, 1, 8},
        {4, 9, 5}, 
        {2, 4, 11}, 
        {6, 2, 10}, 
        {8, 6, 7}, 
        {9, 8, 1},

        // 5 faces around vertex 3
        {3, 9, 4}, 
        {3, 4, 2}, 
        {3, 2, 6}, 
        {3, 6, 8}, 
        {3, 8, 9}
    };
}