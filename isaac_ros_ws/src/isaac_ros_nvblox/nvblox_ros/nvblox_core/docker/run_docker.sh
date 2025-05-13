#!/bin/bash
set -ex


while getopts ":ha:" OPTION; do
    case $OPTION in

        a)
            ADDITIONAL_DOCKER_ARGS=$OPTARG
            ;;
        h | *)
            echo "Run the nvblox docker"
            echo "Usage:"
            echo "run_docker.sh -a "additional_docker_args""
            echo "run_docker.sh -h"
            echo ""
            echo "  -a Additional arguments passed to docker run."
            echo "  -h help (this output)"
            exit 0
            ;;
    esac
done


# This portion of the script will only be executed *inside* the docker when
# this script is used as entrypoint further down. It will setup an user account for
# the host user inside the docker s.t. created files will have correct ownership.
if [ -f /.dockerenv ]
then
    set -euxo pipefail

    # Make sure that all shared libs are found. This should normally not be needed, but resolves a
    # problem with the opencv installation. For unknown reaosns, the command doesn't bite if placed
    # at the end of the dockerfile
    ldconfig

    # Add the group of the user. User/group ID of the host user are set through env variables when calling docker run further down.
    groupadd --force --gid "$DOCKER_RUN_GROUP_ID" "$DOCKER_RUN_GROUP_NAME"

    # Re-add the user
    userdel "$DOCKER_RUN_USER_NAME" || true
    useradd --no-log-init \
            --uid "$DOCKER_RUN_USER_ID" \
            --gid "$DOCKER_RUN_GROUP_NAME" \
            --groups sudo \
            --shell /bin/bash \
            $DOCKER_RUN_USER_NAME
    chown $DOCKER_RUN_USER_NAME /home/$DOCKER_RUN_USER_NAME

    # Change the root user password (so we can su root)
    echo 'root:root' | chpasswd
    echo "$DOCKER_RUN_USER_NAME:root" | chpasswd

    # Allow sudo without password
    echo "$DOCKER_RUN_USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers


    set +x

    GREEN='\033[0;32m'
    IGREEN='\033[0;92m'
    NO_COLOR='\033[0m'

    echo -e "${GREEN}********************************************************"
    echo -e "* ${IGREEN}NVBLOX DEV DOCKER"
    echo -e "${GREEN}********************************************************"
    echo -e ${NO_COLOR}
    # Change into the host user and start interactive session
    su $DOCKER_RUN_USER_NAME
    exit
fi

CONTAINER_NAME=nvblox_deps
DATASETS_FOLDER=$HOME/datasets

docker build --network=host -t $CONTAINER_NAME . -f docker/Dockerfile.deps

# Remove any exited containers.
if [ "$(docker ps -a --quiet --filter status=exited --filter name=$CONTAINER_NAME)" ]; then
    docker rm $CONTAINER_NAME > /dev/null
fi

# If container is running, attach to it, otherwise start
if [ "$( docker container inspect -f '{{.State.Running}}' $CONTAINER_NAME)" = "true" ]; then
  echo "Container already running. Attaching."
  docker exec -it $CONTAINER_NAME su $(id -un)

else
    DOCKER_RUN_ARGS+=("--name" "$CONTAINER_NAME"
                      "--privileged"
                      "--net=host"
                      "--gpus" 'all,"capabilities=compute,utility,graphics"'
                      "-v" ".:/workspaces/nvblox"
                      "-v" "$DATASETS_FOLDER:/datasets"
                      "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
                      "-v" "$HOME/.Xauthority"
                      "--env" "DISPLAY"
                      "--env" "DOCKER_RUN_USER_ID=$(id -u)"
                      "--env" "DOCKER_RUN_USER_NAME=$(id -un)"
                      "--env" "DOCKER_RUN_GROUP_ID=$(id -g)"
                      "--env" "DOCKER_RUN_GROUP_NAME=$(id -gn)"
                      "--entrypoint" "/workspaces/nvblox/docker/run_docker.sh"
                      "--workdir" "/workspaces/nvblox"
                 )
    if [ -n "${ADDITIONAL_DOCKER_ARGS}" ]; then
        DOCKER_RUN_ARGS+=($ADDITIONAL_DOCKER_ARGS)
    fi

    docker run "${DOCKER_RUN_ARGS[@]}" --interactive --rm --tty "$CONTAINER_NAME"
fi
