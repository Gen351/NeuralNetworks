#pragma once

#include <vector>
#include "vec3.h"

inline std::vector<Vec3> getCatVertices() {
    return { 
        // --- HEAD (0 - 7) ---
        {.x = -0.25, .y =  0.35, .z =  0.25}, // 0: Head Front-Left-Top
        {.x =  0.25, .y =  0.35, .z =  0.25}, // 1: Head Front-Right-Top
        {.x =  0.25, .y =  0.35, .z = -0.15}, // 2: Head Back-Right-Top
        {.x = -0.25, .y =  0.35, .z = -0.15}, // 3: Head Back-Left-Top
        {.x = -0.20, .y =  0.05, .z =  0.30}, // 4: Head Front-Left-Bot
        {.x =  0.20, .y =  0.05, .z =  0.30}, // 5: Head Front-Right-Bot
        {.x =  0.20, .y =  0.05, .z = -0.10}, // 6: Head Back-Right-Bot
        {.x = -0.20, .y =  0.05, .z = -0.10}, // 7: Head Back-Left-Bot

        // --- SNOUT & EARS (8 - 10) ---
        {.x =  0.00, .y =  0.15, .z =  0.45}, // 8: Snout / Nose Tip
        {.x = -0.20, .y =  0.70, .z =  0.10}, // 9: Left Ear Tip
        {.x =  0.20, .y =  0.70, .z =  0.10}, // 10: Right Ear Tip

        // --- BODY (11 - 18) ---
        {.x = -0.30, .y =  0.10, .z = -0.15}, // 11: Body Front-Left-Top
        {.x =  0.30, .y =  0.10, .z = -0.15}, // 12: Body Front-Right-Top
        {.x =  0.25, .y =  0.15, .z = -0.90}, // 13: Body Back-Right-Top
        {.x = -0.25, .y =  0.15, .z = -0.90}, // 14: Body Back-Left-Top
        {.x = -0.25, .y = -0.35, .z = -0.15}, // 15: Body Front-Left-Bot
        {.x =  0.25, .y = -0.35, .z = -0.15}, // 16: Body Front-Right-Bot
        {.x =  0.20, .y = -0.30, .z = -0.85}, // 17: Body Back-Right-Bot
        {.x = -0.20, .y = -0.30, .z = -0.85}, // 18: Body Back-Left-Bot

        // --- PAWS / LEGS (19 - 26) ---
        {.x = -0.22, .y = -0.80, .z = -0.10}, // 19: Front Left Paw Front
        {.x = -0.12, .y = -0.80, .z = -0.20}, // 20: Front Left Paw Back
        {.x =  0.12, .y = -0.80, .z = -0.20}, // 21: Front Right Paw Back
        {.x =  0.22, .y = -0.80, .z = -0.10}, // 22: Front Right Paw Front
        {.x = -0.18, .y = -0.75, .z = -0.80}, // 23: Back Left Paw Front
        {.x = -0.12, .y = -0.75, .z = -0.90}, // 24: Back Left Paw Back
        {.x =  0.12, .y = -0.75, .z = -0.90}, // 25: Back Right Paw Back
        {.x =  0.18, .y = -0.75, .z = -0.80}, // 26: Back Right Paw Front

        // --- TAIL (27 - 28) ---
        {.x =  0.00, .y =  0.35, .z = -1.15}, // 27: Tail Mid
        {.x =  0.00, .y =  0.65, .z = -1.30}  // 28: Tail Tip
    };
}

inline std::vector<std::vector<int>> getCatFaces() {
    return {
        // --- HEAD ---
        {0, 1, 2, 3},       // Head Top
        {0, 1, 8},          // Head Snout Upper
        {0, 8, 4},          // Head Snout Left
        {1, 5, 8},          // Head Snout Right
        {4, 8, 5},          // Head Snout Lower
        {2, 3, 7, 6},       // Head Back
        {3, 0, 4, 7},       // Head Left Side
        {1, 2, 6, 5},       // Head Right Side
        {4, 5, 6, 7},       // Head Bottom

        // --- EARS ---
        {0, 9, 3},          // Left Ear Outer
        {0, 1, 9},          // Left Ear Front
        {3, 9, 2},          // Left Ear Back
        {1, 10, 2},         // Right Ear Outer
        {1, 0, 10},         // Right Ear Front
        {2, 10, 3},         // Right Ear Back

        // --- BODY ---
        {11, 12, 16, 15},   // Body Front
        {13, 14, 18, 17},   // Body Back
        {14, 13, 12, 11},   // Body Top
        {15, 16, 17, 18},   // Body Bottom
        {14, 11, 15, 18},   // Body Left
        {12, 13, 17, 16},   // Body Right

        // --- LEGS ---
        {15, 19, 20},       // Front Left Leg Outer
        {15, 11, 19},       // Front Left Leg Front
        {15, 20, 18},       // Front Left Leg Back
        {16, 22, 21},       // Front Right Leg Outer
        {16, 12, 22},       // Front Right Leg Front
        {16, 21, 17},       // Front Right Leg Back
        {18, 23, 24},       // Back Left Leg Outer
        {18, 15, 23},       // Back Left Leg Front
        {18, 24, 17},       // Back Left Leg Back
        {17, 26, 25},       // Back Right Leg Outer
        {17, 16, 26},       // Back Right Leg Front
        {17, 25, 18},       // Back Right Leg Back

        // --- TAIL ---
        {13, 14, 27},       // Tail Base
        {14, 28, 27},       // Tail Left Segment
        {13, 27, 28}        // Tail Right Segment
    };
}