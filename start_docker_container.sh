#!/bin/bash

# Container name
IMAGE_NAME=vsc-real-wbc-7d93562fde6e368633e5e388f71bfc9a07efa3e3ee6d012ae187c8e42a317a8f-uid
# IMAGE_NAME=vsc-real-wbc

docker run -it --rm \
  --name wbc-real-container \
  --network=host \
  --runtime=nvidia \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/shm:/dev/shm \
  -v $HOME/.zsh_history:/home/real/.zsh_history \
  -v $HOME/CompanionDoggy:/home/real/CompanionDoggy \
  -v /tmp:/tmp \
  -w /home/real \
  $IMAGE_NAME \
  zsh -c 'source ~/.zshrc; conda activate robot; cd ~/CompanionDoggy; python insta360_line_following_advanced.py' #exec zsh -l'
  # -c conda activate robot; echo test 
  # zsh -l -c conda activate robot; cd ~/CompanionDoggy; python insta360_line_following_advanced.py
