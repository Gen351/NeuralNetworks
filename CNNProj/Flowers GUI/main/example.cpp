#include <iostream>

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


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

    float speed = 0.0f;
    float distance = 0.0f;

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
        ImGui::SliderFloat("Speed", &speed, 0.0f, 5.0f); 

        // Creates another slider for the Z distance
        ImGui::SliderFloat("Distance", &distance, 1.0f, 20.0f);

        // Switch mesh
        if (ImGui::Button("Click Me")) {
            // Logic
        }
        ImGui::End(); // Closes the window declaration



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
