local params = std.extVar("__ksonnet/params").components["dotaservice"];
{
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "labels": {
            "app": "agent-val",
            "job": params.jobname,
        },
        "name": params.jobname + "-agent-val"
    },
    "spec": {
        "replicas": 1,
        "selector": {
            "matchLabels": {
                "app": "agent-val",
                "job": params.jobname
            }
        },
        "template": {
            "metadata": {
                "labels": {
                    "app": "agent-val",
                    "job": params.jobname,
                }
            },
            "spec": {
                "containers": [
                    {
                        "args": [
                            "--ip",
                            params.jobname + "-rmq.default.svc.cluster.local",
                            "--max-dota-time",
                            std.toString(params.max_dota_time),
                            "--validation",
                            "1",
                            "--log-dir",
                            std.toString(params.expname) + "/" + std.toString(params.jobname) + "/val"
                        ],
                        "command": [
                            "python3.7",
                            "agent.py"
                        ],
                        "env": [
                            {
                                "name": "GOOGLE_APPLICATION_CREDENTIALS",
                                "value": "/etc/gcp/sa_credentials.json"
                            }
                        ],
                        "image": "gcr.io/dotaservice-225201/dotaclient:" + params.dotaclient_image_tag,
                        "name": "agent",
                        "resources": {
                            "requests": {
                                "cpu": "700m"
                            }
                        },
                        "volumeMounts": [
                            {
                                "mountPath": "/etc/gcp",
                                "name": "gcs-secret",
                                "readOnly": true
                            }
                        ]
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
                    },
                    {
                        "name": "gcs-secret",
                        "secret": {
                            "items": [
                                {
                                    "key": "sa_json",
                                    "path": "sa_credentials.json"
                                }
                            ],
                            "secretName": "gcs-admin-secret"
                        }
                    }
                ]
            }
        }
    }
}