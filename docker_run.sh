

docker run -v ${PWD}:/dy -v /media/hdd/dataset/open_dataset:/datasets  -d --runtime=nvidia -it --rm --name yolo3_docker yolo3_tf:1
