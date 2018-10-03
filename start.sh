#!/bin/bash
jupyter notebook --allow-root --NotebookApp.token='' --NotebookApp.password='' &
tensorboard --logdir logs --port 6006 --debugger_port 6064 &

cleanup ()
{
kill -s SIGTERM $!
exit 0
}

trap cleanup SIGINT SIGTERM

while [ 1 ]
do
    sleep 60 &
    wait $!
done
