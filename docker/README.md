
From the repo root, run:
```sh
docker build -t dotaclient:latest . -f docker/Dockerfile
docker build -t rmq:3.7-plugins . -f docker/Dockerfile-rmq
```