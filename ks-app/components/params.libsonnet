{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    dotaservice: {
      agents: 8,
      dotaclient_image_tag: 'latest',
      dotaservice_image_tag: '0.3.8',
      entropy_coef: '5e-4',
      epochs: 1,
      expname: 'exp14',
      jobname: 'job1',
      learning_rate: '2e-5',
      max_dota_time: 600,
      min_seq_per_epoch: 1024,
      optimizers: 1,
      pretrained_model: '',
      rollout_size: 9999,
      seq_len: 16,
      use_latest_weights_prob: '1.0',
      vf_coef: '0.5',
    },
  },
}