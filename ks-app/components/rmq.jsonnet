local params = std.extVar("__ksonnet/params").components["dotaservice"];

[{
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {
        "labels": {
            "app": "rmq",
            "job": params.jobname,
        },
        "name": params.jobname + '-rmq',
    },
    "spec": {
        "ports": [
            {
                "name": "http",
                "port": 15672,
                "protocol": "TCP",
                "targetPort": 15672
            },
            {
                "name": "amqp",
                "port": 5672,
                "protocol": "TCP",
                "targetPort": 5672
            }
        ],
        "selector": {
            "app": "rmq",
            "job": params.jobname,
        },
        "type": "ClusterIP"
    }
},
{
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "name": params.jobname + '-rmq',
        "labels": {
            "app": "rmq",
            "job": params.jobname,
        }
    },
    "spec": {
        "replicas": 1,
        "selector": {
            "matchLabels": {
                "app": "rmq",
                "job": params.jobname
            }
        },
        "template": {
            "metadata": {
                "labels": {
                    "app": "rmq",
                    "job": params.jobname,
                }
            },
            "spec": {
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "cloud.google.com/gke-preemptible",
                                            "operator": "DoesNotExist"
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                },
                "containers": [
                    {
                        "image": "gcr.io/dotaservice-225201/rmq:3.7-plugins",
                        "name": "rmq",
                        "ports": [
                            {
                                "containerPort": 15672,
                                "name": "http",
                                "protocol": "TCP"
                            },
                            {
                                "containerPort": 5672,
                                "name": "amqp",
                                "protocol": "TCP"
                            }
                        ],
                        "resources": {
                            "requests": {
                                "cpu": "200m",
                                "memory": "1024Mi"
                            }
                        }
                    }
                ]
            }
        }
    }
}]