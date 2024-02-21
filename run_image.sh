#!/bin/sh
docker run --rm --gpus all -p {port1}:{port2} -v $(pwd):/src/notebooks -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes -e NB_UID={UID} -e PASSWORD=password --user root mattck/stab
