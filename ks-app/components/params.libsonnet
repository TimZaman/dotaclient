{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 22,
      batch_size: 8,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.8',
      entropy_coef: '0.0',
      epochs: 1,
      expname: 'exp7',
      jobname: 'job2',
      learning_rate: '2e-5',
      max_dota_time: 600,
      optimizers: 1,
      pretrained_model: '',
      rollout_size: 9999,
      seq_len: 1400,
      seq_per_epoch: 16,
      use_latest_weights_prob: '0.8',
      vf_coef: '0.0',
    },
  },
}