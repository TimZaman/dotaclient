{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 14,
      batch_size: 8,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.7',
      epochs: 4,
      expname: 'exp5',
      jobname: 'job1',
      learning_rate: '1e-6',
      entropy_coef: '0.001',
      vf_coef: '0.05',
      max_dota_time: 600,
      optimizers: 1,
      pretrained_model: '',
      rollout_size: 9999,
      seq_len: 256,
      seq_per_epoch: 32,
    },
  },
}