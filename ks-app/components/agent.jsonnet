local params = std.extVar("__ksonnet/params").components["dotaservice"];
{
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "labels": {
            "app": "agent",
            "job": params.jobname,
        },
        "name": params.jobname + "-agent"
    },
    "spec": {
        "replicas": params.agents,
        "selector": {
            "matchLabels": {
                "app": "agent",
                "job": params.jobname
            }
        },
        "template": {
            "metadata": {
                "labels": {
                    "app": "agent",
                    "job": params.jobname,
                }
            },
            "spec": {
                "containers": [
                    {
                        "args": [
                            "--ip",
                            params.jobname + "-rmq.default.svc.cluster.local",
                            "--rollout-size",
                            std.toString(params.rollout_size),
                            "--max-dota-time",
                            std.toString(params.max_dota_time),
                            "--use-latest-weights-prob",
                            std.toString(params.use_latest_weights_prob)
                        ],
                        "command": [
                            "python3.7",
                            "agent.py"
                        ],
                        "image": "gcr.io/dotaservice-225201/dotaclient:" + params.dotaclient_image_tag,
                        "name": "agent",
                        "resources": {
                            "requests": {
                                "cpu": "700m"
                            }
                        }
                    },
                    {
                        "args": [
                            "--action-path",
                            "/ramdisk"
                        ],
                        "command": [
                            "python3.7",
                            "-m",
                            "dotaservice"
                        ],
                        "image": "gcr.io/dotaservice-225201/dotaservice:" + params.dotaservice_image_tag,
                        "name": "dotaservice",
                        "ports": [
                            {
                                "containerPort": 13337
                            }
                        ],
                        "resources": {
                            "requests": {
                                "cpu": "800m",
                                "memory": "1536Mi"
                            }
                        },
                        "volumeMounts": [
                            {
                                "mountPath": "/ramdisk",
                                "name": "ramdisk"
                            }
                        ]
                    }
                ],
                "nodeSelector": {
                    "cloud.google.com/gke-preemptible": "true"
                },
                "volumes": [
                    {
                        "emptyDir": {
                            "medium": "Memory"
                        },
                        "name": "ramdisk"
                    }
                ]
            }
        }
    }
}