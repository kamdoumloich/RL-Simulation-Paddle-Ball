Dieses Repository enthält RL-Simulationsdateien im Falle eines Hybridsystems. Hierbei handelt es sich um das Spiel Paddle-Ball. In diesem Spiel muss ein Agent lernen, einen Ball so lange wie möglich mit einem Paddel zu hüpfen, wobei er stets weist, dass der Ball immer über einem festen Schwellenwert von 5 bleiben muss. 
Die Simulationen umfassen den Anwendungsfall von zwei Arten von Funktionsapproximationen:
- Diskretisierung (*bouncing_ball_DQN.py*)
- NN (*bouncing_ball.py*, *agent_nn.py*)
Die Umgebung sowie die Plotting-Funktionen können im Datei *bouncing_env.py* und *plotting_functions.py* gefunden werden  

Die Folder *nn_pictures* und *q_learning_pictures* beinhalten die Bilder der Dynamik für beide Systeme.
