local params = std.extVar("__ksonnet/params").components["dotaservice"];
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
                "app": "rmq"
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
                                "memory": "1408Mi"
                            }
                        }
                    }
                ]
            }
        }
    }
}