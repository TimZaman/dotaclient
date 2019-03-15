{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 32,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.9',
      entropy_coef: '6e-4',
      epochs: 1,
      expname: 'exp14',
      jobname: 'job4',
      learning_rate: '3e-5',
      max_dota_time: 600,
      min_seq_per_epoch: 1280,
      optimizers: 1,
      pretrained_model: 'exp13/job8/model_000004400.pt',
      rollout_size: 9999,
      seq_len: 16,
      use_latest_weights_prob: '1.0',
      vf_coef: '0.5',
    },
  },
}