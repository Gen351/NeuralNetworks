#include <iostream>
#include <numbers> // for std::numbers::pi
#include <cmath> // for cos(), sin()
#include <vector>

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


int main(int argc, char* argv[]) {
    if (!glfwInit()) return -1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // 1. Init window
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Sample", NULL, NULL);
    if (!window) return -1;
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);


    float sample1 = 0.0f;
    float sample2 = 0.0f;


    // 2. MAIN APPLICATION LOOP
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        // Start frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ===START: Display the graphichs here ================ //
        
        


        
        // ===END: Display the graphichs here ================== //





        // ===START: Creates a new floating window ============= //
        
        ImGui::Begin("3D Settings");
        
        // Creates a float slider 
        // Parameters: (<Label>, <Pointer to value>, <Min value>, <Max value>);
        ImGui::SliderFloat("Rotation Speed", &sample1, 0.0f, 5.0f); 
        
        // Creates another float slider
        ImGui::SliderFloat("Camera Z Distance", &sample2, 1.0f, 20.0f);
        
        // Creates text
        ImGui::Text("Sample1: %f", sample1);
        // Creates text
        ImGui::Text("Sample2: %f", sample2);
        ImGui::End(); // Closes the window declaration

        // ===END: Creates a new floating window =============== //
        


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

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
