# Ideas for Projects

## 1. Interactive Canvas MNIST Digit Recognizer: 
> ImGui drawing canvas feeding mouse pixels live into custom CNN engine with real-time probability bar charts.


## 2. Terminal-Based Char-Level Recurrent NN:
> Pure TUI language generator predicting and completing console text input character by character.


## 3. Image Autoencoder & Latent Space Canvas (I don't understand):
> C++ CNN autoencoder that compresses images into a latent vector and allows slider-based latent manipulation inside ImGui.


## 4. Self-Playing Platformer Agent (Raylib Platformer):
> Frame-by-frame platformer state vector (tile grid + velocity) passed to an RL network training an agent to run and jump to the goal.


## 5. Real-time Object Tracker via Webcam:
> Capture video frames using OpenCV, pass downsampled regions through custom CNN, and draw bounding boxes in ImGui.


## 6. Chess Engine (Minimax + NNUE): 
#### Done, not that good, can defeat beginners
> Use a TUI/Ncurses interface. Evaluate board positions with an efficient neural network architecture trained on raw board state matrices.


## 7. Connect Four (MCTS + Custom Value Neural Net):
> Build a simple TUI or ImGui board using Monte Carlo Tree Search guided by a custom policy/value neural network.


## 8. Smaller Backprop:
> Derive the inputs to lower rank(?)(from a vector of 10 inputs -> a vector of 2 inputs), the same as from d/dx(x^2) = 2x, x is now only rank 1.
> - IDK if that's how it works.


## 9. Training a Rubix Cube solver:
> First: start with a solved cube -> move 1 -> let the AI guess what move it should do to solve it(revese the move) -> backprop
> - Problem: 
>   - How to do the backprop
>   - Cube has fixed moves(right?) so the output vector should be fixed therefore you can caclulate loss, therefore backprop
>   - But doing that, it won't know the "sequence" of moves to solve the cube.
> - Idea of a solution:
>   - Maybe make it guess the "sequence" of moves, then accumulate the loss, then backprop(?), but that math wouldn't make sense.



# TO DO:
> - What is RL Network.
>	- Learn RNN.
>	- Make smaller CNN.
>	- Need data for training.



# Current:
> - 