#!/bin/bash
nice docker build --pull --rm --tag jupyter:build .
#!/bin/bash
docker rm jupyterlab -f
docker run --name=jupyterlab \
    --restart=always \
    -p 8888:8888 \
    --link mongodb \
    jupyter:build