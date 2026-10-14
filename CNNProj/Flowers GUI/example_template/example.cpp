#include <iostream>
#include <numbers> // for std::numbers::pi
#include <cmath> // for cos(), sin()
#include <vector>

#include "vec3.h"
#include "penger.h"
#include "icosahedron.h"
#include "cat.h"

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// =======================================
// EXAMPLE
// =======================================
ImVec2 Project3D(const Vec3& p, ImVec2 screen_center, double focal_length) {
    if (p.z <= 0.1f) return ImVec2(-9999, -9999);

    double x_2d = screen_center.x + (p.x / p.z) * focal_length;
    double y_2d = screen_center.y - (p.y / p.z) * focal_length;

    return ImVec2(x_2d, y_2d);
}
// =======================================
// EXAMPLE END
// =======================================

// =======================================
// EXAMPLE
// =======================================
void Draw3DMesh(const std::vector<Vec3>& vertices, const std::vector<std::vector<int>>& faces, bool cull=false) {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 center = ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
    double focal_length = 400.0f;

    // 1. Project all 3D vertices to 2D screen positions once
    std::vector<ImVec2> points2D(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        points2D[i] = Project3D(vertices[i], center, focal_length);
    }

    // 2. Iterate through each face and draw its outer edges
    for (const auto& face : faces) {
        size_t faceSize = face.size();
        if (faceSize < 2) continue; // Skip invalid faces

        for (size_t i = 0; i < faceSize; ++i) {
            int idxA = face[i];
            int idxB = face[(i + 1) % faceSize]; // Wrap around to close the polygon loop

            // Bounds check
            if (idxA < vertices.size() && idxB < vertices.size()) {
                ImVec2 pA = points2D[idxA];
                ImVec2 pB = points2D[idxB];

                // Skip drawing if either point is clipped behind camera (-9999)
                if (pA.x != -9999 && pB.x != -9999) {
                    draw_list->AddLine(pA, pB, IM_COL32(0, 255, 200, 255), 2.0f);
                }
            }
        }
    }
}
// =======================================
// EXAMPLE END
// =======================================

// =======================================
// EXAMPLE
// =======================================
void rotateXZ(Vec3& p, double angle) {
    const double cosAngle = std::cos(angle);
    const double sinAngle = std::sin(angle);

    const double oldX = p.x;
    p.x = (oldX * cosAngle) - (p.z * sinAngle);
    p.z = (oldX * sinAngle) + (p.z * cosAngle);
}
// =======================================
// EXAMPLE END
// =======================================

// =======================================
// EXAMPLE
// =======================================
enum MESH {
    PENGER = 0,
    ICOSAHEDRON = 1,
    CAT = 2,
};
// =======================================
// EXAMPLE END
// =======================================
int main(int argc, char* argv[]) {
    if (!glfwInit()) return -1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "ImGui Animated 3D vertices", NULL, NULL);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // =======================================
    // EXAMPLE
    // =======================================
    const double fastForward = (argc >= 2) ? std::stof(argv[1]) : 1.0;
    const double FPS = 144;
    const double dt = fastForward / FPS;
    const double pi = std::numbers::pi;

    float rotateSpeed = 0.1;
    float cameraDist = (argc >= 3) ? std::stof(argv[2]) : 3.0;

    MESH currentMesh = MESH::PENGER;

    std::vector<Vec3> vertices;
    std::vector<std::vector<int>> faces;
    vertices = getPengerVertices();
    faces = getPengerFaces();
    currentMesh = MESH::PENGER;

    double currentAngle = 0.0;
    // =======================================
    // EXAMPLE END
    // =======================================

    // 2. MAIN APPLICATION LOOP
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        // Start frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();


        // 2. Put this inside your main rendering loop (between ImGui::NewFrame and ImGui::Render)
        ImGui::Begin("3D Settings"); // Creates a new floating window

        // Creates a float slider named "Rotation Speed" linked to your variable
        // Parameters: Label, Pointer to value, Min value, Max value
        ImGui::SliderFloat("Rotation Speed", &rotateSpeed, 0.0f, 5.0f); 

        // Creates another slider for the Z distance
        ImGui::SliderFloat("Camera Z Distance", &cameraDist, 1.0f, 20.0f);

        // Switch mesh
        if (ImGui::Button("Change Mesh")) {
            if(currentMesh == MESH::PENGER) {
                vertices = getIcosahedronVertices();
                faces = getIcosahedronFaces();
                currentMesh = MESH::ICOSAHEDRON;
            } else if(currentMesh == MESH::ICOSAHEDRON) {
                vertices = getCatVertices();
                faces = getCatFaces();
                currentMesh = MESH::CAT;
            } else if(currentMesh == MESH::CAT) {
                vertices = getPengerVertices();
                faces = getPengerFaces();
                currentMesh = MESH::PENGER;
            }
        }
        
        ImGui::Text("Current Mesh Vertices: %zu", vertices.size());
        ImGui::Text("Current Mesh Faces: %zu", faces.size());
        
        ImGui::End(); // Closes the window declaration


        // =======================================
        // EXAMPLE
        // =======================================
        // 3. UPDATE POSITION ONCE PER FRAME
        currentAngle += rotateSpeed * pi * dt;
        std::vector<Vec3> renderedVertices = vertices;

        for(Vec3& p : renderedVertices) {
            rotateXZ(p, currentAngle);
            p.z += cameraDist;
            if(currentMesh == MESH::ICOSAHEDRON) p.z -= 3;
        }
        // =======================================
        // EXAMPLE END
        // =======================================



        // =======================================
        // EXAMPLE
        // =======================================
        // 4. DRAW vertices ONCE PER FRAME
        Draw3DMesh(renderedVertices, faces);
        // =======================================
        // EXAMPLE END
        // =======================================

        // Finalize ImGui frame
        ImGui::Render();

        // Clear screen and draw OpenGL data
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.12f, 0.15f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
