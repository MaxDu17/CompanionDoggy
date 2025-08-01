#!/bin/bash

# Container name
# IMAGE_NAME=vsc-real-wbc-7d93562fde6e368633e5e388f71bfc9a07efa3e3ee6d012ae187c8e42a317a8f-uid
IMAGE_NAME=vsc-real-wbc


# # Check if the container is already running
# if ! docker ps --format '{{.Names}}' | grep -qw "$CONTAINER_NAME"; then
#   # Run the container in detached mode with a pseudo-tty, interactive shell
#   docker run -dit --name "$CONTAINER_NAME" /bin/bash
# fi

# # Open a terminal (gnome-terminal) and attach it to the container
# gnome-terminal -- docker exec -it "$CONTAINER_NAME" /bin/bash

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
  -w /home/real \
  --user 1003:1005 \
  $IMAGE_NAME \
  zsh -l

  # -v $(pwd):/home/real/$(basename $(pwd)) \
