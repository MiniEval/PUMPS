docker build -t mo_pumps .
xhost +local:docker
~/../rerun --port 1998 &
docker run -it --rm \
  -v "$PWD/../datasets/processed":/datasets/processed \
  -v "$PWD/../datasets/CMU":/datasets/CMU \
  -v "$PWD/../datasets/LaFAN1":/datasets/LaFAN1 \
  -v "$PWD/../datasets/100STYLE":/datasets/100STYLE \
  -v "$PWD/../datasets/Human3.6M":/datasets/Human3.6M \
  -v "$PWD/../datasets/AMASS":/datasets/AMASS \
  -v "$PWD/../datasets/smpl":/datasets/smpl \
  -v "$PWD/checkpoints":/app/checkpoints \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e CUDA_VISIBLE_DEVICES=0,1,2 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --shm-size=8g \
  --gpus all \
  --ipc=host \
  --network host \
  mo_pumps \
  python3 $1
#   /bin/bash
