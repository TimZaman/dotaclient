# DotaClient on K8s

* Distributed Agents self-playing Dota 2.
* Experience/Model Broker (rmq).
* Distributed Optimizer (PyTorch)

## Prerequisites

* Kubeflow's PyTorch Operator
* Google Cloud Storage (GCS)
* https://github.com/TimZaman/DotaService and build the dotaservice into a Docker image
* Build the dotaservice docker image
* Build the rabbitmq docker image

```
kubectl create -f manifests/rmq.yaml
kubectl create -f manifests/dotaservice.yaml
kubectl create -f manifests/optimizer.yaml
```
