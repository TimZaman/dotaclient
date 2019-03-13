local params = std.extVar("__ksonnet/params").components["dotaservice"];

local worker(replicas) = {
    "replicas": std.toString(replicas),
    "restartPolicy": "OnFailure",
    "template": {
        "metadata": {
            "labels": {
                "app": "optimizer",
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
                    "args": [
                        "--ip",
                        params.jobname + "-rmq.default.svc.cluster.local",
                        "--epochs",
                        std.toString(params.epochs),
                        "--min-seq-per-epoch",
                        std.toString(params.min_seq_per_epoch),
                        "--seq-len",
                        std.toString(params.seq_len),
                        "--learning-rate",
                        std.toString(params.learning_rate),
                        "--entropy-coef",
                        std.toString(params.entropy_coef),
                        "--vf-coef",
                        std.toString(params.vf_coef)
                    ],
                    "command": [
                        "python3.7",
                        "optimizer.py"
                    ],
                    "image": "gcr.io/dotaservice-225201/dotaclient:" + params.dotaclient_image_tag,
                    "name": "pytorch",
                    "resources": {
                        "requests": {
                            "cpu": "600m",
                            "memory": "4096Mi"
                        }
                    }
                }
            ]
        }
    }
};

{
    "apiVersion": "kubeflow.org/v1beta1",
    "kind": "PyTorchJob",
    "metadata": {
        "name": params.jobname + "-optimizer",
        "labels": {
            "app": "optimizer",
            "job": params.jobname,
        }
    },
    "spec": {
        "pytorchReplicaSpecs": {
            "Master": {
                "replicas": 1,
                "restartPolicy": "OnFailure",
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "optimizer",
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
                                "args": [
                                    "--ip",
                                    params.jobname + "-rmq.default.svc.cluster.local",
                                    "--epochs",
                                    std.toString(params.epochs),
                                    "--seq-per-epoch",
                                    std.toString(params.seq_per_epoch),
                                    "--batch-size",
                                    std.toString(params.batch_size),
                                    "--seq-len",
                                    std.toString(params.seq_len),
                                    "--learning-rate",
                                    std.toString(params.learning_rate),
                                    "--entropy-coef",
                                    std.toString(params.entropy_coef),
                                    "--vf-coef",
                                    std.toString(params.vf_coef),
                                    "--log-dir",
                                    std.toString(params.expname) + "/" + std.toString(params.jobname),
                                ] + if params.pretrained_model == '' then [] else [
                                    '--pretrained-model', params.pretrained_model ,
                                ],
                                "command": [
                                    "python3.7",
                                    "optimizer.py"
                                ],
                                "env": [
                                    {
                                        "name": "GOOGLE_APPLICATION_CREDENTIALS",
                                        "value": "/etc/gcp/sa_credentials.json"
                                    }
                                ],
                                "image": "gcr.io/dotaservice-225201/dotaclient:" + params.dotaclient_image_tag,
                                "name": "pytorch",
                                "resources": {
                                    "requests": {
                                        "cpu": "600m",
                                        "memory": "4096Mi"
                                    }
                                },
                                "volumeMounts": [
                                    {
                                        "mountPath": "/etc/gcp",
                                        "name": "gcs-secret",
                                        "readOnly": true
                                    }
                                ]
                            }
                        ],
                        "volumes": [
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
            }, [if params.optimizers > 1 then 'Worker']: worker(params.optimizers-1),
        },
        "terminationGracePeriodSeconds": 30
    }
}
