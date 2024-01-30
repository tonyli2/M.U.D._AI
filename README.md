# M.U.D._AI

<p align="center">
  <a href="https://www.youtube.com/watch?v=RBzfRr98-vI">Video Demonstration of Machine Learning Robot</a>
</p>

<p align="center">
  <img width="700" alt="board" src="https://github.com/tonyli2/M.U.D._AI/assets/73260050/88ca2067-cb48-41a3-bfd8-53abec6a5c4b">
</p>

# Skills/Tools Used
- Python
- Machine Learning (Imitation Learning)
- Computer Vision (OpenCV)
- ROS
- Gazebo
- PID Control Loop

# Project Background

This project emerged as the collaborative effort of Tony Li and Hunter Ma for the Engineering Physics 353 (ENPH 353) Machine Learning Competition. ENPH 353 is a course that introduces the fundamental concepts of machine learning to students via a hands-on, self-directed, competition. In this contest, a randomized famous individual has been murdered, and students are tasked with creating a Gazebo Simulated autonomous robot detective, which not only self-navigates the simulation environment but interacts with it as well to solve this mystery. The detective will be tasked with driving around the simulation roadways (while avoiding pedestrians and other vehicles) and will read sign boards off the side of the road which contain hints as to who committed the murder and why. Once the detective robot reads the text on each signboard, it will communicate what it believes it reads to a score tracker. The score tracker will compare the predicted text to the actual ground truth clue on each board and distribute points for correct answers. In the end, the score tracker will utilize the Chat-GPT API to generate an overarching summary of the murder case with the clues gathered.

<p align="center">
  <img width="501" alt="first" src="https://github.com/tonyli2/M.U.D._AI/assets/73260050/24e9016f-11e4-4aa8-929d-52409104d5bc">
</p>

# Project Organization

For team workflow and management/organization of the project tasks and objectives, we used [Linear](https://linear.app/). With Linear, the team was able to coordinate and parallelize the workflow at every part of the project by creating tickets and visually representing the work needed for each task. The figure below showcases what the project board looked like a few days before the competition:

<p align="center">
  <img width="688" alt="second" src="https://github.com/tonyli2/M.U.D._AI/assets/73260050/fa6fe6c3-881d-4a69-a7e1-abcf9ae9ac56">
</p>

# Software Architecture

The figure below shows the overall software architecture of this project. The pink shapes represent all four preset ROS topics that were used to communicate with the detective car. The two essential components, autonomous driving, and clue board detection form the left and right graphs respectively.
<p align="center">
  <img width="688" alt="third" src="https://github.com/tonyli2/M.U.D._AI/assets/73260050/21df9792-43b0-4dc0-b174-8bd896633044">
</p>
