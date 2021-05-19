#PORT=8000 && docker run -p 5000:${PORT} -e PORT=${PORT} api_flask 
#!/bin/bash
cd app
docker rm api_flask -f
PORT=5000 && 
docker run -it --name=api_flask \
    --volume /$(pwd):/app \
    -p 5000:${PORT}\
    -e PORT=${PORT} api_flask
    #-d -p 5000:5000 api_flask
    # python_processing:build \
    # python test_preprocessing.py \