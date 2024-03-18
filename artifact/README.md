# How to Run

Steps to build the containerized enviroment and run the scripts. 

```bash
cd rescience-submission-73 # repository root

docker build -t re_76 -f artifact/dockerfile .

docker run --rm -v ${PWD}:/workspace -it re_76  bash
```


## Inside the Container


```bash

python generation_env.py

python main.py

python parameter_fitting.py

```