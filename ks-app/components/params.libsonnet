{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 50,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.9',
      entropy_coef: '8e-4',
      epochs: 2,
      expname: 'exp16',
      jobname: 'job3',
      learning_rate: '1e-5',
      max_dota_time: 600,
      min_seq_per_epoch: 2048,
      optimizers: 1,
      pretrained_model: 'exp16/job2/model_000002313.pt',
      rollout_size: 9999,
      seq_len: 16,
      use_latest_weights_prob: '1.0',
      vf_coef: '0.5',
    },
  },
}